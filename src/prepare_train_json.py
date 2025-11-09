# src/prepare_train_json.py  (overwrite)
import json
import os

def build_relative_path(subdir, idx, prefix_candidate_list=None):
    """Try several filename patterns and return the first that exists.
    Returns the relative path (subdir/filename) even if file doesn't exist (first candidate),
    to preserve behavior; but prefers an existing file.
    """
    if prefix_candidate_list is None:
        prefix_candidate_list = [None]  # None -> plain "<idx>.png"
    candidates = []
    for pref in prefix_candidate_list:
        if pref is None:
            fname = f"{idx}.png"
        else:
            fname = f"{pref}{idx}.png"
        rel = os.path.join(subdir, fname).replace("\\", "/")
        candidates.append(rel)

    # prefer the first candidate that actually exists
    for rel in candidates:
        if os.path.exists(os.path.join("data", "train_images", rel)):
            return rel

    # if none exist, return the first candidate (so subsequent steps still have a path)
    return candidates[0]

def main():
    fake_json = os.path.join("data", "fake_cifake_preds.json")
    real_json = os.path.join("data", "real_cifake_preds.json")
    output_json = os.path.join("data", "train.json")

    fake_subdir = "fake_cifake_images"
    real_subdir = "real_cifake_images"

    with open(fake_json, "r") as f:
        fake_data = json.load(f)
    with open(real_json, "r") as f:
        real_data = json.load(f)

    combined = []

    # Candidates: try with "fake_" prefix first, then no prefix
    for item in fake_data:
        idx = item.get("index")
        pred = str(item.get("prediction","")).strip().lower()
        if idx is None:
            continue

        rel_path = build_relative_path(fake_subdir, idx, prefix_candidate_list=["fake_", None])
        combined.append({
            "image_path": rel_path,
            "label": 0,
            "target": 0.0 if pred == "fake" else 1.0
        })

    # For real images, try "real_" prefix then no prefix
    for item in real_data:
        idx = item.get("index")
        pred = str(item.get("prediction","")).strip().lower()
        if idx is None:
            continue

        rel_path = build_relative_path(real_subdir, idx, prefix_candidate_list=["real_", None])
        combined.append({
            "image_path": rel_path,
            "label": 1,
            "target": 1.0 if pred == "real" else 0.0
        })

    with open(output_json, "w") as f:
        json.dump(combined, f, indent=4)

    print(f"✅ Combined {len(combined)} entries written to {output_json}")

    # Quick summary of missing files (if any)
    missing = []
    for e in combined:
        full = os.path.join("data", "train_images", e["image_path"])
        if not os.path.exists(full):
            missing.append(e["image_path"])
    if missing:
        print("⚠️ Files not found for the following image_path entries (count):", len(missing))
        print("Example missing paths:", missing[:10])
    else:
        print("✅ All referenced image files exist.")

if __name__ == "__main__":
    main()
