#!/usr/bin/env python3
"""GRPO training of one policy: LoRA on the language model, vision tower frozen,
correctness checker as reward. Settings come from environment variables (see jobs/)."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import hashlib
import json
import os
import sys
import time


SCRIPT_VERSION = "rlvr_train/1.1"


def _cap_arrow_threads(n=2):
    """Keep parquet reads to a couple of threads per task: many tasks reading one file with a
    thread pool each is the access pattern the file-system team flagged."""
    try:
        import pyarrow as pa
        pa.set_cpu_count(n); pa.set_io_thread_count(n)
    except Exception:
        pass


_cap_arrow_threads()

# LoRA targets the language model's attention and MLP projections only.
# Verified against the real checkpoint's safetensors index, not guessed.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

INSTRUCTION = ("Solve the problem step by step, then give the final answer "
               "inside \\boxed{}.")


def env(name, default=None, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def build_dataset(dataset: str, data_root: str, prompt_ids: list[str], processor):
    """HF Dataset of {prompt (chat), image, gold}. Prompts come from the hashed split."""
    from datasets import Dataset
    from rollout_gen import load_items_by_id

    import io
    from PIL import Image as PILImage

    items = load_items_by_id(dataset, prompt_ids, data_root)
    rows = {"prompt": [], "images": [], "gold": [], "prompt_id": []}
    for it in items:
        if not it["images"]:
            continue
        rows["prompt"].append([{
            "role": "user",
            "content": [{"type": "image"},
                        {"type": "text",
                         "text": f"{it['question']}\n\n{INSTRUCTION}"}],
        }])
        # TRL's _tokenize_prompts reads part["image"] from INSIDE the message
        # content, which prepare_multimodal_messages fills from this column. It
        # tries the PLURAL "images" column (a list per example) before the
        # singular one, and the value must be a decoded PIL image: raw bytes and
        # {"bytes":...} dicts both reach the processor undecoded and raise
        # TypeError in fetch_images. Decode eagerly here so nothing depends on
        # when the Image feature happens to decode.
        rows["images"].append(
            [PILImage.open(io.BytesIO(it["images"][0])).convert("RGB")])
        # a multiple-choice gold is a LIST (letter and option text); str() would
        # turn it into "['D', '140']" and the reward would never fire. Carry it
        # as JSON and decode in the reward function.
        g = it["gold"]
        rows["gold"].append(json.dumps(g) if isinstance(g, (list, tuple)) else str(g))
        rows["prompt_id"].append(it["prompt_id"])
    return Dataset.from_dict(rows)


def make_reward_fn(dataset: str):
    """Binary verified-outcome reward -- the SAME verifier the rollouts use.

    Using a different verifier for training than for measurement would make the
    measured coverage a property of the evaluation verifier and the learned
    policy a property of another, so they must be one function.
    """
    import csm_verify

    def reward_verified(completions, **kwargs):
        golds = kwargs.get("gold")
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else (
                c[-1]["content"] if isinstance(c, list) and c else "")
            g = golds[i] if golds is not None else None
            if isinstance(g, str) and g.startswith("[") and g.endswith("]"):
                try:
                    g = json.loads(g)
                except ValueError:
                    pass
            out.append(1.0 if csm_verify.verify(text, g, dataset) else 0.0)
        return out

    reward_verified.__name__ = f"verified_outcome_{dataset}"
    return reward_verified


def write_ancestry(path: str, name: str, model_path: str, ids_file: str,
                   cfg: dict, seed: int) -> None:
    """Emit the record csm_ancestry.py (E0) audits."""
    rec = {
        "name": name,
        "base_checkpoint": os.path.join(model_path, "CSM_PIN.txt"),
        "data_manifest": ids_file,
        "config": {**cfg,
                   "seed": seed, "init_seed": seed,
                   "data_order_seed": seed, "rollout_seed": seed},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)


def main() -> int:
    import torch
    from datasets import Dataset  # noqa: F401  (import cost paid once)
    from peft import LoraConfig
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    seed = env("CSM_SEED", 1, int)
    dataset = env("CSM_DATASET", "geometry3k")
    n_train = env("CSM_N_TRAIN", 400, int)
    k_gen = env("CSM_NUM_GENERATIONS", 8, int)
    max_steps = env("CSM_MAX_STEPS", 200, int)
    lr = env("CSM_LR", 1e-6, float)
    # beta=0 => no KL reference model. TRL only skips the reference when
    # beta == 0.0 OR is_peft_model(model) is true, and the latter is evaluated on
    # what we PASS -- a path string, not a PeftModel -- so with beta>0 it fell
    # through and loaded a SECOND full 7B copy (~15 GiB) as the reference. That
    # was the bulk of two CUDA OOMs. beta=0.0 is also TRL's own default and is
    # standard in current RLVR recipes, which drop the KL penalty.
    beta = env("CSM_BETA", 0.0, float)
    temperature = env("CSM_TEMP", 1.0, float)
    max_completion = env("CSM_MAX_COMPLETION", 512, int)
    save_steps = env("CSM_SAVE_STEPS", 25, int)
    lora_r = env("CSM_LORA_R", 16, int)
    lora_alpha = env("CSM_LORA_ALPHA", 32, int)
    micro_batch = env("CSM_MICRO_BATCH", 1, int)   # in GROUPS, not sequences
    grad_accum = env("CSM_GRAD_ACCUM", 4, int)
    max_pixels = env("CSM_MAX_PIXELS", 512 * 28 * 28, int)
    min_pixels = env("CSM_MIN_PIXELS", 128 * 28 * 28, int)

    data_root = os.environ.get("DATA_DIR", "")
    model_path = env("CSM_MODEL_PATH",
                     os.path.join(data_root, "CSM", "models", "Qwen2.5-VL-7B-Instruct"))
    run_tag = env("CSM_RUN_TAG", "")
    run_name = f"seed{seed}{run_tag}"
    out_dir = env("CSM_OUT_DIR",
                  os.path.join(data_root, "CSM", "runs", f"{dataset}__{run_name}"))
    out_path = os.environ.get("OUT", "rlvr_train.csv")

    ids_file = f"analysis/splits/{dataset}_train_rlvr.txt"
    with open(ids_file) as f:
        pids = [ln.strip() for ln in f if ln.strip()][:n_train]

    cfg = {
        "lr": lr, "beta": beta, "temperature": temperature, "k_gen": k_gen,
        "max_steps": max_steps, "max_completion": max_completion,
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_targets": LORA_TARGETS,
        "grad_accum": grad_accum, "micro_batch": micro_batch, "n_train": len(pids),
        "max_pixels": max_pixels, "min_pixels": min_pixels,
        "trainable_modules": ["llm-lora"], "vision_tower": "frozen", "run_tag": run_tag,
        "script_version": SCRIPT_VERSION,
    }
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    print(f"run={SCRIPT_VERSION} seed={seed} dataset={dataset} n_train={len(pids)} "
          f"k_gen={k_gen} max_steps={max_steps} lr={lr} beta={beta} "
          f"lora_r={lora_r} cfg_hash={cfg_hash} out={out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    write_ancestry(os.path.join(out_dir, "ancestry.json"), run_name,
                   model_path, ids_file, cfg, seed)

    torch.manual_seed(seed)

    # FAIL FAST on hardware that cannot do bf16. Volta (V100, sm_70) has no native
    # bfloat16; running bf16 there produced a CUDA device-side assert 850s into
    # generation, with an async traceback pointing at an unrelated line.
    #
    # Deliberately an ERROR, not a silent fp16 fallback: the paper's central claim
    # is that the parents differ ONLY by the seed. A seed trained in fp16 on a
    # V100 and another in bf16 on an A100 differ by numerics as well, which is an
    # ancestry violation csm_ancestry.py would not catch because dtype is not in
    # the config hash. Pin the hardware instead (run --uniform).
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"progress gpu={name} sm={cap[0]}{cap[1]} bf16_supported={bf16_ok}")
        if not bf16_ok:
            print(f"FAIL gpu={name} (sm_{cap[0]}{cap[1]}) has no native bf16; this run "
                  f"would train in different numerics from a bf16 seed and break the "
                  f"seed-only premise. Re-run on bf16-capable hardware.", file=sys.stderr)
            return 4

    def _mem(tag):
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 2**30
            r = torch.cuda.max_memory_allocated() / 2**30
            print(f"progress mem[{tag}] allocated={a:.2f}GiB peak={r:.2f}GiB")

    _mem("start")
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels,
                                              max_pixels=max_pixels)
    ds = build_dataset(dataset, data_root, pids, processor)
    print(f"progress dataset_built n={len(ds)}")

    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        init_lora_weights=True,          # seeded by torch.manual_seed(seed) above
    )

    args = GRPOConfig(
        output_dir=out_dir,
        seed=seed,
        data_seed=seed,                  # data order is a seed component
        learning_rate=lr,
        beta=beta,
        temperature=temperature,
        num_generations=k_gen,
        max_completion_length=max_completion,
        # MUST be a multiple of num_generations: GRPO requires each batch to hold
        # whole prompt groups, since the advantage is group-relative. TRL rejects
        # auto_find_batch_size for the same reason. So one group per device step.
        per_device_train_batch_size=micro_batch * k_gen,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        # bf16=True above is MIXED-PRECISION TRAINING; it does not change the dtype
        # the checkpoint is loaded in. Without this, TRL calls from_pretrained with
        # no dtype and gets fp32: 7B x 4 bytes = 28.4 GiB of weights, which is why
        # three OOMs showed ~37 GiB allocated REGARDLESS of batch size or beta.
        # KEY NAME MATTERS: transformers renamed torch_dtype -> dtype, and the old
        # key is SILENTLY IGNORED here -- introspection showed 8.34B params all in
        # float32 (31.07 GiB) despite passing torch_dtype. rollout_gen.py already
        # loads with dtype= on this same stack. Verified by assertion below.
        model_init_kwargs={"dtype": "bfloat16"},
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=env("CSM_LOG_STEPS", 10, int),
        use_vllm=False,                  # vLLM is not installed in this env
        report_to=[],
        torch_empty_cache_steps=1,       # release cached blocks between steps
        scale_rewards=True,
    )

    _mem("before_trainer")
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=make_reward_fn(dataset),
        args=args,
        train_dataset=ds,
        processing_class=processor,
        peft_config=peft_cfg,
    )

    # resume from the newest checkpoint if the walltime cap already killed a run
    resume = None
    ckpts = [d for d in os.listdir(out_dir) if d.startswith("checkpoint-")]
    if ckpts:
        resume = os.path.join(out_dir, max(ckpts, key=lambda d: int(d.split("-")[1])))
        print(f"progress resuming_from={resume}")

    _mem("after_trainer")

    # Do not guess again: report what is actually resident. 31 GiB after trainer
    # construction is either fp32 weights (torch_dtype ignored) or two bf16 copies.
    # dtype histogram + a scan for distinct top-level modules distinguishes them.
    try:
        import collections as _c
        hist = _c.Counter()
        total = 0
        for _, prm in trainer.model.named_parameters():
            hist[str(prm.dtype)] += prm.numel()
            total += prm.numel() * prm.element_size()
        print("progress model_param_dtypes=" + json.dumps(
            {k: round(v / 1e9, 3) for k, v in hist.items()}) + " (billions of params)")
        print(f"progress model_param_bytes={total / 2**30:.2f}GiB")
        print(f"progress ref_model_is_none={getattr(trainer, 'ref_model', 'ABSENT') is None}")
        seen = {}
        for attr in ("model", "ref_model", "_wrapped_model", "model_wrapped"):
            m = getattr(trainer, attr, None)
            if m is not None and hasattr(m, "parameters"):
                try:
                    p0 = next(m.parameters())
                    seen[attr] = f"{id(p0)}|{p0.dtype}"
                except StopIteration:
                    seen[attr] = "no-params"
        print("progress module_identities=" + json.dumps(seen))

        # A silently-ignored dtype kwarg doubles memory and OOMs ~10 minutes later,
        # far from its cause. Fail here instead, where the message is unambiguous.
        big = [d for d in hist if "float32" in d and hist[d] > 1e9]
        if big:
            print(f"FAIL model loaded in {big} ({total/2**30:.2f}GiB): the dtype "
                  f"kwarg was not honoured. Expected bfloat16.", file=sys.stderr)
            return 5
    except Exception as e:
        print(f"progress introspection_failed={type(e).__name__}:{str(e)[:120]}")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    trainer.save_model(os.path.join(out_dir, "final"))

    m = getattr(result, "metrics", {}) or {}
    summ = {
        "script_version": SCRIPT_VERSION, "seed": seed, "dataset": dataset,
        "cfg_hash": cfg_hash, "n_train": len(ds), "k_gen": k_gen,
        "max_steps": max_steps, "elapsed_s": round(elapsed, 1),
        "train_loss": m.get("train_loss"),
        "train_runtime": m.get("train_runtime"),
        "out_dir": out_dir,
    }
    with open(out_path, "w") as f:
        f.write(",".join(summ) + "\n")
        f.write(",".join(str(v) for v in summ.values()) + "\n")

    print("RESULT " + " ".join(f"{k}={v}" for k, v in summ.items()))
    print(f"wrote={out_path} adapter={out_dir}/final ancestry={out_dir}/ancestry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
