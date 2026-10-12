"""
Real liveness / presentation-attack-detection experiment on the NUAA
Photograph Imposter Database (Tan et al., ECCV 2010; reference [21]/[22] in
the paper), using the dataset's own official train/test split
(client_train_face.txt / imposter_train_face.txt for training,
client_test_face.txt / imposter_test_face.txt for testing - exactly the
3,491 train / 9,123 test split reported in the paper's Table II).

Method: Local Binary Pattern (LBP) texture histogram + linear SVM. This is
a lightweight, classical, CPU-only baseline appropriate for this
environment (no camera / Android hardware, no GPU). It is NOT the
MobileNetV2 transfer-learning model used by the base paper [1] (Section
8.8) - that distinction is stated explicitly in the results discussion.
The dataset provides exactly the "live camera photo" vs "printed-photo
recapture" distinction needed for a genuine bona-fide vs. presentation-
attack liveness measurement (2D print-attack, matching Attack Level A in
Table VI).
"""
import os
import json
import time
import math
import statistics
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
NUAA_DIR = os.path.join(BASE, "..", "data", "nuaa", "extracted", "Detectedface")
RESULTS_DIR = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 64
LBP_RADIUS = 1
LBP_POINTS = 8
GRID = 4  # 4x4 spatial blocks -> block-wise LBP histograms concatenated,
          # following the standard face-anti-spoofing LBP recipe
          # (Chingovska et al. 2012) rather than one global histogram,
          # which preserves the spatial texture pattern that a single
          # whole-face histogram discards.


def load_list(txt_path, subdir):
    paths = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split(" ")[0].replace("\\", "/")
            paths.append(os.path.join(NUAA_DIR, subdir, rel))
    return paths


def extract_lbp_hist(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    block = IMG_SIZE // GRID
    feats = []
    for by in range(GRID):
        for bx in range(GRID):
            cell = lbp[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            feats.append(hist)
    return np.concatenate(feats)


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo) * 100, min(1.0, hi) * 100)


def main():
    client_train = load_list(os.path.join(NUAA_DIR, "client_train_face.txt"), "ClientFace")
    client_test = load_list(os.path.join(NUAA_DIR, "client_test_face.txt"), "ClientFace")
    imposter_train = load_list(os.path.join(NUAA_DIR, "imposter_train_face.txt"), "ImposterFace")
    imposter_test = load_list(os.path.join(NUAA_DIR, "imposter_test_face.txt"), "ImposterFace")

    print(f"train: {len(client_train)} live + {len(imposter_train)} spoof = {len(client_train)+len(imposter_train)}")
    print(f"test:  {len(client_test)} live + {len(imposter_test)} spoof = {len(client_test)+len(imposter_test)}")

    def build_xy(live_paths, spoof_paths, log_latency=None):
        X, y = [], []
        for p in live_paths:
            t0 = time.perf_counter()
            h = extract_lbp_hist(p)
            t1 = time.perf_counter()
            if h is not None:
                X.append(h); y.append(1)  # 1 = live/bona fide
                if log_latency is not None:
                    log_latency.append((t1 - t0) * 1000.0)
        for p in spoof_paths:
            t0 = time.perf_counter()
            h = extract_lbp_hist(p)
            t1 = time.perf_counter()
            if h is not None:
                X.append(h); y.append(0)  # 0 = spoof/attack
                if log_latency is not None:
                    log_latency.append((t1 - t0) * 1000.0)
        return np.array(X), np.array(y)

    print("Extracting LBP features (train)...")
    X_train, y_train = build_xy(client_train, imposter_train)
    print("Extracting LBP features (test)...")
    test_latency = []
    X_test, y_test = build_xy(client_test, imposter_test, log_latency=test_latency)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=4.0, gamma="scale")
    clf.fit(X_train_s, y_train)

    scores = clf.decision_function(X_test_s)  # higher = more "live"
    preds = (scores >= 0).astype(int)

    # Overall accuracy at default decision boundary
    correct = int((preds == y_test).sum())
    acc = correct / len(y_test) * 100

    # Per-class breakdown (Table XII rows)
    live_mask = y_test == 1
    spoof_mask = y_test == 0
    live_correct = int((preds[live_mask] == 1).sum())
    spoof_correct = int((preds[spoof_mask] == 0).sum())
    n_live = int(live_mask.sum())
    n_spoof = int(spoof_mask.sum())

    bpcer = (n_live - live_correct) / n_live * 100  # bona fide rejected as attack
    apcer = (n_spoof - spoof_correct) / n_spoof * 100  # attack accepted as live

    # EER by sweeping threshold on decision_function scores
    thresholds = np.unique(scores)
    best = None
    for thr in thresholds:
        pred_t = (scores >= thr).astype(int)
        far = ((pred_t[spoof_mask] == 1).sum()) / n_spoof  # spoof accepted as live
        frr = ((pred_t[live_mask] == 0).sum()) / n_live    # live rejected
        diff = abs(far - frr)
        if best is None or diff < best[0]:
            best = (diff, thr, far, frr)
    _, eer_thr, far_eer, frr_eer = best
    eer = (far_eer + frr_eer) / 2 * 100
    hter = (apcer / 100 + bpcer / 100) / 2 * 100

    result = {
        "method": f"Block-wise Local Binary Pattern histograms ({GRID}x{GRID} grid, P={LBP_POINTS}, R={LBP_RADIUS}, uniform) + RBF-kernel SVM (classical CPU baseline; NOT the MobileNetV2 transfer-learning model of the base paper)",
        "dataset": "NUAA Photograph Imposter Database (Tan et al. 2010), official face-detected release, official train/test split",
        "train_n": {"live": len(client_train), "spoof": len(imposter_train), "total": len(client_train) + len(imposter_train)},
        "test_n": {"live": n_live, "spoof": n_spoof, "total": len(y_test)},
        "accuracy_pct": round(acc, 2),
        "accuracy_95ci_pct": list(round(x, 2) for x in wilson_ci(correct, len(y_test))),
        "bpcer_pct": round(bpcer, 2),
        "apcer_pct": round(apcer, 2),
        "acer_pct": round((bpcer + apcer) / 2, 2),
        "hter_pct_at_default_threshold": round(hter, 2),
        "eer_pct": round(eer, 2),
        "live_detection_rate_pct": round(live_correct / n_live * 100, 2),
        "spoof_rejection_rate_pct": round(spoof_correct / n_spoof * 100, 2),
        "live_correct_count": live_correct,
        "spoof_correct_count": spoof_correct,
        "latency_ms_feature_extraction": {
            "n": len(test_latency),
            "mean": round(statistics.mean(test_latency), 4),
            "sd": round(statistics.pstdev(test_latency), 4),
            "min": round(min(test_latency), 4),
            "max": round(max(test_latency), 4),
        },
    }
    print(json.dumps(result, indent=2))
    with open(os.path.join(RESULTS_DIR, "liveness_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved results/liveness_results.json")


if __name__ == "__main__":
    main()
