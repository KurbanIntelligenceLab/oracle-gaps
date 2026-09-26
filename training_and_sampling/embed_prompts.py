#!/usr/bin/env python3
"""Image-aware prompt embeddings from the base model, used by the embedding routers."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, io, os, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rollout_gen as RG   # noqa: E402  (loaders, chat template, model loader)

SCRIPT_VERSION = "embed_prompts/1.0"


def main() -> int:
    t0 = time.time()
    dataset, split = os.environ["ITEM"].split("|")
    data_root = os.environ["DATA_DIR"]
    model_path = RG.env("CSM_MODEL_PATH", os.path.join(data_root, "CSM", "models", "Qwen2.5-VL-7B-Instruct"))
    max_pixels = RG.env("CSM_MAX_PIXELS", 1024 * 28 * 28, int); min_pixels = RG.env("CSM_MIN_PIXELS", 256 * 28 * 28, int)
    ids = [l.strip() for l in open(os.environ["CSM_PROMPT_IDS"]) if l.strip()]
    items = RG.load_items_by_id(dataset, ids, data_root)
    print(f"{SCRIPT_VERSION} dataset={dataset} split={split} n={len(items)} model={model_path}")
    model, processor = RG.load_model(model_path, max_pixels, min_pixels, None)
    dev = next(model.parameters()).device
    embs, out_ids, n_tok = [], [], []
    with torch.no_grad():
        for i, item in enumerate(items):
            msgs = RG.build_messages(item)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in item["images"]]
            inputs = processor(text=[text], images=pil if pil else None, return_tensors="pt")
            inputs = {kk: (vv.to(dev) if hasattr(vv, "to") else vv) for kk, vv in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1][0].float()                    # (T, d), image and text tokens
            embs.append(h.mean(0).cpu().numpy().astype(np.float32)); out_ids.append(item["prompt_id"]); n_tok.append(int(h.shape[0]))
            if (i + 1) % 40 == 0:
                print(f"progress {i+1}/{len(items)} elapsed={time.time()-t0:.0f}s")
    E = np.stack(embs)
    out_csv = os.environ["OUT"]; out_npz = os.path.splitext(out_csv)[0] + ".npz"
    np.savez_compressed(out_npz, prompt_ids=np.array(out_ids), embeddings=E, dataset=dataset, split=split, model=model_path, script_version=SCRIPT_VERSION)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["script_version", "dataset", "split", "n_prompts", "dim", "mean_tokens", "npz", "elapsed_s"])
        w.writerow([SCRIPT_VERSION, dataset, split, len(out_ids), E.shape[1], round(float(np.mean(n_tok)), 1), os.path.basename(out_npz), round(time.time() - t0, 1)])
    print(f"done dataset={dataset} split={split} n={len(out_ids)} dim={E.shape[1]} mean_tokens={np.mean(n_tok):.0f} elapsed_s={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
