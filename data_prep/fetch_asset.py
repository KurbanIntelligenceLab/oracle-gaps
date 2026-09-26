#!/usr/bin/env python3
"""Download one Hugging Face asset (see assets.txt) into $DATA_DIR and record its revision."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import os
import sys
import time

SCRIPT_VERSION = "fetch_asset/1.0"


def dir_bytes(path: str) -> tuple[int, int]:
    """(total_bytes, n_files), following the symlinks snapshot_download makes."""
    total = 0
    n = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.stat(fp).st_size  # stat follows symlinks
                n += 1
            except OSError:
                pass
    return total, n


def main() -> int:
    item = os.environ.get("ITEM", "").strip()
    out_path = os.environ.get("OUT", "fetch_asset.csv")
    data_dir = os.environ.get("DATA_DIR", "")
    task_id = os.environ.get("TASK_ID", "0")

    if not item:
        print("FAIL reason=no_ITEM", file=sys.stderr)
        return 2

    parts = [p.strip() for p in item.split("|")]
    if len(parts) != 3:
        print(f"FAIL reason=bad_manifest_line item={item!r}", file=sys.stderr)
        return 2
    kind, name, repo_id = parts
    repo_type = "model" if kind == "model" else "dataset"
    sub = "models" if kind == "model" else "datasets"
    dest = os.path.join(data_dir, "CSM", sub, name)

    print(f"run={SCRIPT_VERSION} task={task_id} kind={kind} name={name} "
          f"repo={repo_id} dest={dest}")

    from huggingface_hub import HfApi, snapshot_download

    t0 = time.time()
    status = "ok"
    note = ""
    sha = ""
    try:
        api = HfApi()
        info = api.repo_info(repo_id, repo_type=repo_type, timeout=60)
        sha = info.sha or ""
        print(f"progress resolved repo={repo_id} sha={sha}")

        os.makedirs(dest, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=sha or None,
            local_dir=dest,
            max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        )
    except Exception as e:
        status = "FAILED"
        note = f"{type(e).__name__}:{str(e)[:200]}"
        print(f"FAIL name={name} repo={repo_id} err={note}", file=sys.stderr)

    elapsed = time.time() - t0
    nbytes, nfiles = dir_bytes(dest) if os.path.isdir(dest) else (0, 0)

    # record the pinned revision next to the data, so the snapshot is
    # self-describing even if this shard is lost
    if status == "ok":
        try:
            with open(os.path.join(dest, "CSM_PIN.txt"), "w") as f:
                f.write(f"repo_id={repo_id}\nrepo_type={repo_type}\nrevision={sha}\n"
                        f"fetched_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                        f"script={SCRIPT_VERSION}\n")
        except OSError:
            pass

    with open(out_path, "w") as f:
        f.write("task_id,kind,name,repo_id,repo_type,revision,status,"
                "bytes,gb,n_files,elapsed_s,dest,note\n")
        f.write(f'{task_id},{kind},{name},{repo_id},{repo_type},{sha},{status},'
                f'{nbytes},{nbytes/1e9:.3f},{nfiles},{elapsed:.1f},{dest},"{note}"\n')

    print(f"result name={name} status={status} gb={nbytes/1e9:.3f} files={nfiles} "
          f"revision={sha} elapsed_s={elapsed:.1f}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
