#!/usr/bin/env python3
"""Sample k responses per prompt from a policy (base, or base plus adapter) and score them."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import io
import json
import os
import sys
import time


SCRIPT_VERSION = "rollout_gen/1.2"


def _cap_arrow_threads(n=2):
    """Keep parquet reads to a couple of threads per task: many tasks reading one file with a
    thread pool each is the access pattern the file-system team flagged."""
    try:
        import pyarrow as pa
        pa.set_cpu_count(n); pa.set_io_thread_count(n)
    except Exception:
        pass


_cap_arrow_threads()


def env(name: str, default=None, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return cast(v)


# ------------------------------------------------------------ dataset adapters
PATTERNS = {
    "chartqa": {"train": "train-*", "val": "val-*", "test": "test-*"},
    "geometry3k": {"train": "train-*", "validation": "validation-*",
                   "val": "validation-*", "test": "test-*"},
    # second-dataset candidates: only testmini carries public answers, and the
    # study's hashed val/test are subsets of it (ids read '<ds>-testmini-<row>')
    "mathvista": {"testmini": "testmini-*"},
    "mathverse": {"testmini": "testmini"},
}
# parquet location relative to the dataset root; MathVerse keeps it at the root
DIRS = {"chartqa": "data", "geometry3k": "data", "mathvista": "data", "mathverse": ""}
# text/image columns each adapter reads. Confirmed against inspect_datasets.
COLS = {
    "chartqa": ["image", "query", "label"],
    "geometry3k": ["images", "problem", "answer"],
    "mathvista": ["decoded_image", "query", "question", "answer", "choices",
                  "question_type", "metadata"],
    "mathverse": ["image", "question", "answer", "question_type",
                  "problem_index", "problem_version", "metadata"],
}


def _split_files(dataset: str, split: str, data_root: str):
    import glob
    root = os.path.join(data_root, "CSM", "datasets", dataset)
    pat = PATTERNS[dataset][split]
    sub = DIRS.get(dataset, "data")
    base = os.path.join(root, sub) if sub else root
    files = sorted(glob.glob(os.path.join(base, f"{pat}.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet for {dataset}/{split} under {base}")
    return files


def load_items_by_id(dataset: str, prompt_ids, data_root: str):
    """Resolve explicit prompt_ids of the form '<dataset>-<official_split>-<row>'.

    This is the path the STUDY uses: prompts come from the hashed splits file, so
    every model is scored on exactly the same prompt set and nothing is sampled
    ad hoc. Returned in the order given.
    """
    import pyarrow.parquet as pq

    want: dict[str, list[tuple[int, str]]] = {}
    for pid in prompt_ids:
        try:
            ds, split, row = pid.rsplit("-", 2)
            row_i = int(row)
        except (ValueError, AttributeError):
            raise ValueError(f"malformed prompt_id {pid!r}")
        if ds != dataset:
            raise ValueError(f"prompt_id {pid!r} is not from dataset {dataset!r}")
        want.setdefault(split, []).append((row_i, pid))

    by_pid: dict[str, dict] = {}
    for split, pairs in want.items():
        files = _split_files(dataset, split, data_root)
        counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
        starts, acc = [], 0
        for c in counts:
            starts.append(acc)
            acc += c
        cols = _cols(dataset, files[0])

        per_file: dict[int, list[tuple[int, str]]] = {}
        for row_i, pid in sorted(pairs):
            if row_i >= acc:
                raise ValueError(f"{pid}: row {row_i} beyond split size {acc}")
            fi = max(i for i, s in enumerate(starts) if s <= row_i)
            per_file.setdefault(fi, []).append((row_i - starts[fi], pid))

        for fi in sorted(per_file):
            local = per_file[fi]
            t = pq.read_table(files[fi], columns=cols).take([l for l, _ in local])
            d = t.to_pydict()
            for j, (_, pid) in enumerate(local):
                by_pid[pid] = _row_to_item(dataset, d, j, pid)

    return [by_pid[p] for p in prompt_ids]


def _cols(dataset: str, sample_file: str) -> list[str]:
    """Columns to read, restricted to those the parquet actually has."""
    import pyarrow.parquet as pq
    if dataset not in COLS:
        raise KeyError(f"no adapter for dataset {dataset!r}")
    have = {f.name for f in pq.ParquetFile(sample_file).schema_arrow}
    cols = [c for c in COLS[dataset] if c in have]
    missing = [c for c in COLS[dataset][:3] if c not in have]   # image + text + gold
    if missing:
        raise KeyError(f"{dataset}: parquet lacks required columns {missing}; has {sorted(have)}")
    return cols


def _img_bytes(x):
    """HF image cells arrive as {'bytes','path'} structs, raw bytes, or None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x.get("bytes")
    return x if isinstance(x, (bytes, bytearray)) else None


def _mc_golds(letter: str | None, choices: list[str] | None, answer: str | None):
    """Accept either the option letter or the option text on multiple choice.

    The verifier takes a list of golds. A model told to box its final answer
    writes whichever form it reasons to; scoring only one of them would count a
    correct '140 degrees' wrong against gold 'D', which is a verifier bias, not a
    policy failure.
    """
    golds = []
    if answer is not None and str(answer).strip():
        golds.append(str(answer).strip())
    if letter and choices and letter.strip().upper() in "ABCDEFGH":
        i = ord(letter.strip().upper()) - ord("A")
        if 0 <= i < len(choices) and choices[i] is not None:
            golds.append(str(choices[i]).strip())
    if answer is not None and choices and str(answer).strip() in [str(c).strip() for c in choices]:
        golds.append(chr(ord("A") + [str(c).strip() for c in choices].index(str(answer).strip())))
    seen, out = set(), []
    for g in golds:
        if g not in seen:
            seen.add(g); out.append(g)
    return out if len(out) > 1 else (out[0] if out else answer)


def _row_to_item(dataset: str, d: dict, j: int, prompt_id: str) -> dict:
    if dataset == "chartqa":
        img = d["image"][j]
        imgs = [img["bytes"] if isinstance(img, dict) else img]
        q = d["query"][j]
        gold = d["label"][j]
    elif dataset == "geometry3k":
        raw = d["images"][j] or []
        imgs = [(im["bytes"] if isinstance(im, dict) else im) for im in raw]
        q = (d["problem"][j] or "").replace("<image>", "").strip()
        gold = d["answer"][j]
    elif dataset == "mathvista":
        # 'query' is the benchmark's own prompt: hint + question + choices
        b = _img_bytes(d["decoded_image"][j])
        imgs = [b] if b else []
        q = (d.get("query", [None])[j] or d["question"][j] or "").strip()
        choices = d.get("choices", [None])[j]
        ans = d["answer"][j]
        gold = _mc_golds(None, list(choices) if choices else None, ans) \
            if (d.get("question_type", [None])[j] == "multi_choice") else ans
    elif dataset == "mathverse":
        b = _img_bytes(d["image"][j])
        imgs = [b] if b else []
        q = (d["question"][j] or "").strip()
        ans = d["answer"][j]
        if d.get("question_type", [None])[j] == "multi-choice":
            import re as _re
            choices = {}
            for line in q.splitlines():
                m = _re.match(r"^\s*([A-H])\s*[:.)]\s*(.+?)\s*$", line)
                if m:
                    choices[m.group(1)] = m.group(2)
            letter = str(ans).strip().upper() if ans is not None else None
            gold = [x for x in [letter, choices.get(letter)] if x] or ans
        else:
            gold = ans
    else:
        raise KeyError(f"no adapter for dataset {dataset!r}")
    return {"prompt_id": prompt_id, "question": q, "gold": gold, "images": imgs}


def load_items(dataset: str, split: str, data_root: str, n_items: int, seed: int):
    """Return [{prompt_id, question, gold, image_bytes:[...]}] deterministically.

    Selection is a seeded permutation over the split, so the same n_items are
    drawn for every model -- per-prompt comparison across seeds requires an
    identical prompt set (csm_rollouts.validate enforces this). PILOT path only;
    the study resolves prompts through load_items_by_id from the hashed splits.
    """
    import random
    import pyarrow.parquet as pq

    files = _split_files(dataset, split, data_root)

    cols = _cols(dataset, files[0])

    # A split can span several parquet files (ChartQA's train is 3). Index over
    # the CONCATENATION in sorted filename order, so a prompt_id denotes the same
    # item for every model and every run. Reading one file would silently shrink
    # the pool and break cross-model prompt alignment.
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    n_avail = sum(counts)
    idx = list(range(n_avail))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:min(n_items, n_avail)])

    # map global indices back to (file, local index), then read only the files needed
    starts, acc = [], 0
    for c in counts:
        starts.append(acc)
        acc += c

    per_file: dict[int, list[int]] = {}
    for g in idx:
        fi = max(i for i, s in enumerate(starts) if s <= g)
        per_file.setdefault(fi, []).append(g - starts[fi])

    chunks = []
    for fi in sorted(per_file):
        t = pq.read_table(files[fi], columns=cols)
        chunks.append(t.take(per_file[fi]))
    tbl = chunks[0] if len(chunks) == 1 else __import__("pyarrow").concat_tables(chunks)

    d = tbl.to_pydict()
    # prompt_id is stable across models and runs: dataset-officialsplit-rowindex
    return [_row_to_item(dataset, d, j, f"{dataset}-{split}-{row_i:06d}")
            for j, row_i in enumerate(idx)]


INSTRUCTION = ("Solve the problem step by step, then give the final answer "
               "inside \\boxed{}.")


def build_messages(item):
    content = []
    for _ in item["images"]:
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"{item['question']}\n\n{INSTRUCTION}"})
    return [{"role": "user", "content": content}]


# ------------------------------------------------------------------ the model
def load_model(model_path: str, max_pixels: int, min_pixels: int,
               adapter_path: str | None = None):
    """Load the backbone, optionally with a trained LoRA adapter on top.

    The seed policies ARE base + adapter, so scoring them means loading the same
    pinned base and attaching the adapter written by rlvr_train. Keeping one
    loader for base and seeds guarantees every arm is scored through an identical
    code path -- the per-prompt comparison across members is meaningless
    otherwise.
    """
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    model = None
    errors = []
    for cls_name in ("Qwen2_5_VLForConditionalGeneration",
                     "AutoModelForImageTextToText",
                     "AutoModelForVision2Seq"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_path, dtype=torch.bfloat16,
                                        device_map="auto")
            print(f"progress loaded_with={cls_name}")
            break
        except Exception as e:                       # try the next class
            errors.append(f"{cls_name}:{type(e).__name__}:{str(e)[:120]}")
    if model is None:
        raise RuntimeError("could not load model; tried " + " | ".join(errors))

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()      # fold LoRA in: no adapter overhead at generate
        print(f"progress adapter_merged={adapter_path}")

    model.eval()
    return model, processor


def main() -> int:
    import torch
    from PIL import Image
    import csm_verify

    dataset = env("CSM_DATASET", "chartqa")
    split = env("CSM_SPLIT", "val")
    n_items = env("CSM_N_ITEMS", 200, int)
    k = env("CSM_K", 16, int)
    temperature = env("CSM_TEMP", 1.0, float)
    top_p = env("CSM_TOP_P", 1.0, float)
    max_new = env("CSM_MAX_NEW_TOKENS", 512, int)
    batch_prompts = env("CSM_BATCH_PROMPTS", 2, int)
    model_name = env("CSM_MODEL_NAME", "base")
    seed = env("CSM_SEED", 0, int)
    sample_seed = env("CSM_SAMPLE_SEED", 12345, int)
    max_pixels = env("CSM_MAX_PIXELS", 1024 * 28 * 28, int)
    min_pixels = env("CSM_MIN_PIXELS", 256 * 28 * 28, int)
    save_raw = env("CSM_SAVE_RAW", 1, int)

    data_root = os.environ.get("DATA_DIR", "")
    model_path = env("CSM_MODEL_PATH",
                     os.path.join(data_root, "CSM", "models",
                                  "Qwen2.5-VL-7B-Instruct"))
    out_path = os.environ.get("OUT", "rollout_gen.csv")
    task_id = os.environ.get("TASK_ID", "0")

    print(f"run={SCRIPT_VERSION} model={model_name} dataset={dataset} split={split} "
          f"n_items={n_items} k={k} temp={temperature} max_new={max_new} "
          f"max_pixels={max_pixels} seed={seed} task={task_id}")

    torch.manual_seed(seed)

    # STUDY path: prompts come from the hashed splits file, so every model is
    # scored on an identical prompt set. PILOT path: a seeded subsample.
    ids_file = env("CSM_PROMPT_IDS")
    if ids_file:
        with open(ids_file) as fh:
            pids = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        if n_items and n_items > 0:
            pids = pids[:n_items]
        items = load_items_by_id(dataset, pids, data_root)
        print(f"progress loaded_items={len(items)} source=prompt_ids:{ids_file}")
    else:
        items = load_items(dataset, split, data_root, n_items, sample_seed)
        print(f"progress loaded_items={len(items)} source=subsample:{split}")

    adapter_path = env("CSM_ADAPTER_PATH")
    model, processor = load_model(model_path, max_pixels, min_pixels, adapter_path)
    dev = next(model.parameters()).device
    print(f"progress device={dev} cuda={torch.cuda.is_available()} "
          f"n_gpu={torch.cuda.device_count()}")

    raw_dir = os.path.join(data_root, "CSM", "rollouts_raw",
                           f"{model_name}__{dataset}__{split}")
    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)
    raw_fh = open(os.path.join(raw_dir, f"shard_{task_id}.jsonl"), "w") if save_raw else None

    rows = []
    tot_gen_tokens = 0
    tot_correct = 0
    tot_draws = 0
    t_start = time.time()
    last_report = t_start
    oom_retries = 0

    for i, item in enumerate(items):
        msgs = build_messages(item)
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in item["images"]]

        inputs = processor(text=[text], images=pil if pil else None,
                           return_tensors="pt")
        inputs = {kk: (vv.to(dev) if hasattr(vv, "to") else vv)
                  for kk, vv in inputs.items()}
        in_len = inputs["input_ids"].shape[1]

        completions = []
        gen_tokens = 0
        remaining = k
        chunk = max(1, min(k, batch_prompts * k))
        while remaining > 0:
            take = min(chunk, remaining)
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new,
                        num_return_sequences=take,
                        pad_token_id=processor.tokenizer.pad_token_id
                        or processor.tokenizer.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_retries += 1
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                print(f"progress oom_backoff new_chunk={chunk}")
                continue
            seqs = out[:, in_len:]
            gen_tokens += int((seqs != processor.tokenizer.pad_token_id).sum().item()) \
                if processor.tokenizer.pad_token_id is not None else int(seqs.numel())
            completions.extend(processor.tokenizer.batch_decode(
                seqs, skip_special_tokens=True))
            remaining -= take

        n_correct = 0
        verdicts = []
        for c in completions:
            good, rule = csm_verify.verify_verbose(c, item["gold"], dataset)
            n_correct += int(good)
            verdicts.append(rule)

        tot_correct += n_correct
        tot_draws += len(completions)
        tot_gen_tokens += gen_tokens

        rows.append({
            "prompt_id": item["prompt_id"], "model": model_name, "split": split,
            "dataset": dataset, "n_draws": len(completions), "n_correct": n_correct,
            "gen_tokens": gen_tokens,
        })

        if raw_fh is not None:
            raw_fh.write(json.dumps({
                "prompt_id": item["prompt_id"], "model": model_name,
                "dataset": dataset, "split": split,
                "question": item["question"], "gold": item["gold"],
                "completions": completions, "verdicts": verdicts,
            }) + "\n")

        # coarse progress: at most every 60s, never per item
        now = time.time()
        if now - last_report > 60 or i == len(items) - 1:
            el = now - t_start
            done = i + 1
            print(f"progress {done}/{len(items)} ({100*done/len(items):.0f}%) "
                  f"elapsed_s={el:.0f} tok_per_s={tot_gen_tokens/max(el,1e-9):.1f} "
                  f"items_per_s={done/max(el,1e-9):.3f} "
                  f"mean_rate={tot_correct/max(tot_draws,1):.4f} "
                  f"eta_s={(len(items)-done)*el/max(done,1):.0f}")
            last_report = now

    if raw_fh is not None:
        raw_fh.close()

    elapsed = time.time() - t_start
    mean_rate = tot_correct / max(tot_draws, 1)
    rates = [r["n_correct"] / max(r["n_draws"], 1) for r in rows]
    n_zero = sum(1 for r in rates if r == 0.0)
    n_one = sum(1 for r in rates if r == 1.0)
    in_band = sum(1 for r in rates if 0.05 <= r <= 0.30)

    with open(out_path, "w") as f:
        f.write("prompt_id,model,split,dataset,n_draws,n_correct,gen_tokens\n")
        for r in rows:
            f.write(f"{r['prompt_id']},{r['model']},{r['split']},{r['dataset']},"
                    f"{r['n_draws']},{r['n_correct']},{r['gen_tokens']}\n")

    # summary sidecar -- kept OUT of the shard so the shard stays a clean table
    summ = {
        "script_version": SCRIPT_VERSION, "model": model_name, "dataset": dataset,
        "adapter": adapter_path or "none",
        "split": split, "n_items": len(rows), "k": k, "temperature": temperature,
        "max_new_tokens": max_new, "max_pixels": max_pixels, "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "tokens_per_s": round(tot_gen_tokens / max(elapsed, 1e-9), 2),
        "items_per_s": round(len(rows) / max(elapsed, 1e-9), 4),
        "total_gen_tokens": tot_gen_tokens, "total_draws": tot_draws,
        "mean_rate": round(mean_rate, 5),
        "frac_rate_zero": round(n_zero / max(len(rates), 1), 4),
        "frac_rate_one": round(n_one / max(len(rates), 1), 4),
        "frac_in_band_0.05_0.30": round(in_band / max(len(rates), 1), 4),
        "oom_backoffs": oom_retries,
    }
    with open(out_path.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)

    print("RESULT " + " ".join(f"{k2}={v}" for k2, v in summ.items()))
    print(f"histogram_rate_deciles=" + ",".join(
        str(sum(1 for r in rates if d / 10 <= r < (d + 1) / 10 or
                (d == 9 and r == 1.0))) for d in range(10)))
    print(f"wrote={out_path} rows={len(rows)} raw={raw_dir if save_raw else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
