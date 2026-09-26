"""Run both oracle-gap tests on five simulated policies that are identical by construction.

All policies share each prompt's success probability, so any oracle gap is sampling
noise. Replace V with your own stored verdicts: V[i, m, j] = 1 if response j of
policy m to prompt i is correct.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from nulls_extra import both_nulls  # noqa: E402

rng = np.random.default_rng(1)
n_prompts, n_policies, k = 240, 5, 16
p = rng.beta(0.6, 1.4, size=n_prompts)
V = (rng.random((n_prompts, n_policies, k)) < p[:, None, None]).astype(np.uint8)

tau = 0.10
res = both_nulls(V, k, np.random.default_rng(0), n_perm=2000, taus=(tau,))
for null in ("redeal", "margin"):
    r = res[tau]["L"][null]
    print(f"{null:7s} oracle gap {r['obs']:.3f}  null mean {r['null_mean']:.3f}  "
          f"excess {r['excess']:+.3f}  p = {r['p']:.3f}")
