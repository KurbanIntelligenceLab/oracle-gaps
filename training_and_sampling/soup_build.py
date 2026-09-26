#!/usr/bin/env python3
"""Weight-averaging baseline: one adapter equal to the mean of the seeds' LoRA updates."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, json, os, shutil, sys, time

import torch
from safetensors.torch import load_file, save_file

SCRIPT_VERSION = "soup_build/1.1"


def main() -> int:
    t0 = time.time()
    ds = os.environ.get("ITEM") or os.environ.get("CSM_DATASET", "geometry3k")
    tag = os.environ.get("CSM_RUN_TAG", "_lr1e5")
    seeds = [int(s) for s in os.environ.get("CSM_SOUP_SEEDS", "1,2,3,4,5").split(",")]
    root = os.path.join(os.environ["DATA_DIR"], "CSM", "runs")
    dirs = [os.path.join(root, f"{ds}__seed{s}{tag}", "final") for s in seeds]
    out_dir = os.path.join(root, f"{ds}__soup{tag}", "final")
    print(f"{SCRIPT_VERSION} dataset={ds} tag={tag} seeds={seeds} out={out_dir}")

    cfgs = [json.load(open(os.path.join(d, "adapter_config.json"))) for d in dirs]
    r, alpha = cfgs[0]["r"], cfgs[0]["lora_alpha"]
    rslora = bool(cfgs[0].get("use_rslora", False))
    mods = sorted(cfgs[0]["target_modules"])
    for d, c in zip(dirs, cfgs):
        got = (c["r"], c["lora_alpha"], bool(c.get("use_rslora", False)), sorted(c["target_modules"]))
        assert got == (r, alpha, rslora, mods), f"adapter config differs at {d}: {got} vs {(r, alpha, rslora, mods)}"
        assert not c.get("rank_pattern") and not c.get("alpha_pattern") and not c.get("use_dora", False), "unsupported LoRA variant"
    scaling = alpha / (r ** 0.5) if rslora else alpha / r
    n = len(dirs)

    weights = [load_file(os.path.join(d, "adapter_model.safetensors")) for d in dirs]
    keys = sorted(weights[0].keys())
    for w in weights[1:]:
        assert sorted(w.keys()) == keys, "adapter tensors differ across seeds"
    a_keys = [k for k in keys if k.endswith("lora_A.weight")]
    soup, dtype = {}, weights[0][a_keys[0]].dtype
    norms = {i: 0.0 for i in range(n)}; norm_soup = 0.0; max_err = 0.0
    for ka in a_keys:
        kb = ka.replace("lora_A.weight", "lora_B.weight")
        A = [w[ka].float() for w in weights]; B = [w[kb].float() for w in weights]
        A_cat = torch.cat(A, dim=0)                                   # (n r, in)
        B_cat = torch.cat([b * (scaling / n) for b in B], dim=1)      # (out, n r)
        soup[ka] = A_cat.to(dtype); soup[kb] = B_cat.to(dtype)
        # exact check on the first module and norms on all
        delta = B_cat @ A_cat
        ref = sum(scaling * (B[i] @ A[i]) for i in range(n)) / n
        max_err = max(max_err, float((delta - ref).abs().max()))
        norm_soup += float(delta.norm() ** 2)
        for i in range(n):
            norms[i] += float((scaling * (B[i] @ A[i])).norm() ** 2)
    norm_soup **= 0.5; norms = {i: v ** 0.5 for i, v in norms.items()}

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(dirs[0]):
        src = os.path.join(dirs[0], f)
        if os.path.isfile(src) and f not in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(src, out_dir)
    cfg = dict(cfgs[0]); cfg["r"] = n * r; cfg["lora_alpha"] = n * r; cfg["use_rslora"] = False
    cfg["rank_pattern"] = {}; cfg["alpha_pattern"] = {}
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)
    save_file(soup, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    json.dump({"script_version": SCRIPT_VERSION, "dataset": ds, "seeds": seeds, "source_dirs": dirs,
               "r": r, "alpha": alpha, "use_rslora": rslora, "scaling": scaling, "r_soup": n * r},
              open(os.path.join(out_dir, "soup_provenance.json"), "w"), indent=2)

    row = dict(script_version=SCRIPT_VERSION, dataset=ds, n_adapters=n, modules=len(a_keys), r=r, alpha=alpha,
               r_soup=n * r, max_abs_err=max_err, delta_norm_soup=round(norm_soup, 4),
               delta_norm_seeds=" ".join(f"{norms[i]:.4f}" for i in range(n)), out_dir=out_dir, elapsed_s=round(time.time() - t0, 1))
    with open(os.environ["OUT"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(" ".join(f"{k}={v}" for k, v in row.items()))
    assert max_err < 1e-3, f"soup does not equal the mean update: max_abs_err={max_err}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
