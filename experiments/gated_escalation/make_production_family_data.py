"""Generate validation surfaces with the PRODUCTION family mix.

The three surfaces used earlier in this campaign (`development`,
`tune_degraded`, `validate_fresh`) are not a valid basis for validating a
change: the latter two share seven family definitions written for this
campaign, deliberately weighted toward degraded acquisitions. That is why plain
gated escalation measured 7 rescued / 0 broken across them and 8 / 5 on the
frozen benchmark. Independent seeds are not independent family composition.

This generates two datasets from `generator.dataset_generator.FAMILIES` — the
**production table**, same 15 families and same per-family counts as the frozen
benchmark — at two fresh seeds. Any change validated here is being tested
against the composition it will actually face.

`cross_generator` has no analogue (it is imported external data, not
generated), so these carry 136 pairs against the frozen benchmark's 156. The
integration gate reads criterion 3 as False when the split is absent, which is
a known artifact already documented for gate exception 2 — not a real finding.

    python -m experiments.gated_escalation.make_production_family_data
"""
from __future__ import annotations

import os

from generator.dataset_generator import FAMILIES, generate_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = {"prodfam_a": 883021, "prodfam_b": 517664,
         "prodfam_c": 240719, "prodfam_d": 661438, "prodfam_e": 105293}


def main() -> None:
    for name, seed in SEEDS.items():
        root = os.path.join(ROOT, "data", name)
        print(f"=== {name} (seed {seed}) -> {root}", flush=True)
        generate_dataset(root, seed=seed, families=FAMILIES, verbose=False)
        n = sum(len([f for f in os.listdir(os.path.join(root, s))
                     if f.endswith("_search.png")])
                for s in os.listdir(root) if os.path.isdir(os.path.join(root, s)))
        print(f"    {n} pairs across {len(os.listdir(root))} splits", flush=True)


if __name__ == "__main__":
    main()
