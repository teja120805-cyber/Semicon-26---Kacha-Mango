import sys, os, json
sys.path.insert(0, "/tmp/driftsense")
sys.path.insert(0, "/tmp/driftsense/experiments/hough_subpatch_voting")
import cv2
import numpy as np
import pandas as pd
from evaluation.evaluate import load_manifest
from evaluation import benchmark, metrics
from harness import localize_subpatch

DATA_ROOT = "/tmp/driftsense/data"
BASELINE_CSV = "/tmp/driftsense/outputs/reports/per_pair_results.csv"

def run_split(split, top_k, beta):
    manifest = load_manifest(DATA_ROOT, split)
    rows = []
    for _, row in manifest.iterrows():
        ref = cv2.imread(row["reference_path"], cv2.IMREAD_UNCHANGED)
        search = cv2.imread(row["search_path"], cv2.IMREAD_UNCHANGED)
        result = localize_subpatch(ref, search, top_k=top_k, beta=beta)
        error_px = float(np.hypot(result.x - row["gt_x"], result.y - row["gt_y"]))
        rows.append({**row.to_dict(), "pred_x": result.x, "pred_y": result.y, "error_px": error_px,
                     "confidence": result.confidence, "ambiguity_ratio": result.ambiguity_ratio,
                     "ambiguous": result.ambiguous, "runtime_s": result.runtime_s, "ranking_mode": "subpatch_exploratory"})
    return pd.DataFrame(rows)

top_k, beta = 8, 0.2
splits = ["development", "validation", "held_out", "challenge", "cross_generator"]
dfs = []
for s in splits:
    df = run_split(s, top_k, beta)
    dfs.append(df)
    print(f"{s}: n={len(df)} acc@5px={(df['error_px']<=5).mean():.3f}")
candidate_df = pd.concat(dfs, ignore_index=True)
candidate_df.to_csv("/tmp/driftsense/experiments/hough_subpatch_voting/outputs/per_pair_results_subpatch_exploratory.csv", index=False)

baseline_df = pd.read_csv(BASELINE_CSV)
merged = baseline_df.sort_values("pair_id").reset_index(drop=True).merge(
    candidate_df.sort_values("pair_id").reset_index(drop=True), on="pair_id", suffixes=("_b","_c"))
merged["dist"] = np.hypot(merged["pred_x_c"]-merged["pred_x_b"], merged["pred_y_c"]-merged["pred_y_b"])
print("pairs with ANY coordinate change:", (merged["dist"]>1e-6).sum(), "/", len(merged))
print("baseline acc@5px:", (baseline_df["error_px"]<=5).mean())
print("exploratory acc@5px:", (candidate_df["error_px"]<=5).mean())

gate = benchmark.run_integration_gate(baseline_df, candidate_df, seeds_agree=True)
with open("/tmp/driftsense/experiments/hough_subpatch_voting/outputs/exploratory_gate_result.json","w") as f:
    json.dump(gate, f, indent=2)
print(json.dumps({"passed": gate["passed"], "criteria": gate["criteria"]}, indent=2))
