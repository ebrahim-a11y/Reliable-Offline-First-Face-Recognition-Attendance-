"""
Real face-verification reproduction on an LFW subset using:
  - YuNet face detector (OpenCV Zoo, Apache-2.0)
  - SFace embedding model (OpenCV Zoo, Apache-2.0; encodes a MobileFaceNet
    instance trained with the SFace loss - see Section 8.6/8.7 of the paper)

Pair construction: genuine = two different images of the same LFW identity;
impostor = one image each from two different identities. This follows the
same same-identity/different-identity definition as the official LFW View 2
protocol, built directly from the extracted subset rather than the official
pairs.txt (whose host was unreachable from this environment - disclosed in
the paper's methodology section).
"""
import cv2
import numpy as np
import os
import json
import random
import math
import time
import statistics
import glob

SEED = 7
random.seed(SEED)

BASE = os.path.dirname(__file__)
SUBSET_DIR = os.path.join(BASE, "..", "data", "lfw_subset", "lfw_funneled")
MODELS_DIR = os.path.join(BASE, "..", "models")
RESULTS_DIR = os.path.join(BASE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(BASE, "..", "data", "lfw_subset", "manifest.json")

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

detector = cv2.FaceDetectorYN.create(
    os.path.join(MODELS_DIR, "yunet.onnx"), "", (320, 320),
    score_threshold=0.7, nms_threshold=0.3, top_k=5000,
)
recognizer = cv2.FaceRecognizerSF.create(
    os.path.join(MODELS_DIR, "sface.onnx"), ""
)


def get_embedding(img_path, detect_latency_log=None, embed_latency_log=None):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    t0 = time.perf_counter()
    _, faces = detector.detect(img)
    t1 = time.perf_counter()
    if detect_latency_log is not None:
        detect_latency_log.append((t1 - t0) * 1000.0)
    if faces is None or len(faces) == 0:
        return None
    # take highest-confidence face
    face = faces[0]
    t0 = time.perf_counter()
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned)
    t1 = time.perf_counter()
    if embed_latency_log is not None:
        embed_latency_log.append((t1 - t0) * 1000.0)
    return feat


def cosine_sim(a, b):
    return float(recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def main():
    detect_lat = []
    embed_lat = []

    print("Computing embeddings for genuine-pair identities...")
    genuine_embeddings = {}  # identity -> list of embeddings
    for ident, fnames in manifest["genuine_identities"].items():
        embs = []
        for fn in fnames:
            path = os.path.join(SUBSET_DIR, ident, fn)
            emb = get_embedding(path, detect_lat, embed_lat)
            if emb is not None:
                embs.append(emb)
        if len(embs) >= 2:
            genuine_embeddings[ident] = embs
    print(f"  usable genuine identities (face detected in >=2 images): {len(genuine_embeddings)}")

    print("Computing embeddings for impostor-pair identities...")
    impostor_embeddings = {}  # identity -> single embedding
    for ident, fnames in manifest["impostor_identities"].items():
        path = os.path.join(SUBSET_DIR, ident, fnames[0])
        emb = get_embedding(path, detect_lat, embed_lat)
        if emb is not None:
            impostor_embeddings[ident] = emb
    print(f"  usable impostor identities: {len(impostor_embeddings)}")

    # Build genuine pairs: one pair per identity (first two usable images)
    genuine_pairs = []
    for ident, embs in genuine_embeddings.items():
        genuine_pairs.append((embs[0], embs[1]))
        if len(embs) >= 4:
            genuine_pairs.append((embs[2], embs[3]))

    # Build impostor pairs: random distinct-identity pairs
    impostor_ids = list(impostor_embeddings.keys())
    random.shuffle(impostor_ids)
    impostor_pairs = []
    for i in range(0, len(impostor_ids) - 1, 2):
        impostor_pairs.append((impostor_embeddings[impostor_ids[i]], impostor_embeddings[impostor_ids[i + 1]]))

    print(f"Genuine pairs: {len(genuine_pairs)}  Impostor pairs: {len(impostor_pairs)}")

    genuine_scores = [cosine_sim(a, b) for a, b in genuine_pairs]
    impostor_scores = [cosine_sim(a, b) for a, b in impostor_pairs]

    # ROC sweep to find EER threshold
    all_scores = sorted(set(genuine_scores + impostor_scores))
    best = None
    for thr in all_scores:
        tp = sum(1 for s in genuine_scores if s >= thr)
        fn = len(genuine_scores) - tp
        tn = sum(1 for s in impostor_scores if s < thr)
        fp = len(impostor_scores) - tn
        far = fp / len(impostor_scores) if impostor_scores else 0
        frr = fn / len(genuine_scores) if genuine_scores else 0
        diff = abs(far - frr)
        if best is None or diff < best[0]:
            best = (diff, thr, far, frr, tp, fn, tn, fp)

    _, tau_star, far, frr, tp, fn, tn, fp = best
    eer = (far + frr) / 2 * 100

    accuracy = (tp + tn) / (len(genuine_scores) + len(impostor_scores)) * 100
    n_pairs = len(genuine_scores) + len(impostor_scores)
    expected_errors_per_6000 = (100 - accuracy) / 100 * 6000

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

    lo, hi = wilson_ci(tp + tn, n_pairs)

    result = {
        "model": "SFace (MobileFaceNet backbone, OpenCV Zoo, Apache-2.0)",
        "detector": "YuNet (OpenCV Zoo, Apache-2.0)",
        "dataset": "LFW-funneled subset (Huang et al. 2007), locally-generated genuine/impostor pairs",
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "n_total_pairs": n_pairs,
        "threshold_tau_star": round(tau_star, 4),
        "accuracy_pct": round(accuracy, 2),
        "accuracy_95ci_pct": [round(lo, 2), round(hi, 2)],
        "eer_pct": round(eer, 2),
        "far_pct": round(far * 100, 2),
        "frr_pct": round(frr * 100, 2),
        "expected_errors_per_6000_pairs": round(expected_errors_per_6000, 1),
        "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "genuine_score_stats": {
            "mean": round(statistics.mean(genuine_scores), 4),
            "sd": round(statistics.pstdev(genuine_scores), 4),
            "min": round(min(genuine_scores), 4),
            "max": round(max(genuine_scores), 4),
        },
        "impostor_score_stats": {
            "mean": round(statistics.mean(impostor_scores), 4),
            "sd": round(statistics.pstdev(impostor_scores), 4),
            "min": round(min(impostor_scores), 4),
            "max": round(max(impostor_scores), 4),
        },
        "latency_ms": {
            "detection": {
                "n": len(detect_lat), "mean": round(statistics.mean(detect_lat), 3),
                "sd": round(statistics.pstdev(detect_lat), 3),
                "min": round(min(detect_lat), 3), "max": round(max(detect_lat), 3),
            },
            "embedding": {
                "n": len(embed_lat), "mean": round(statistics.mean(embed_lat), 3),
                "sd": round(statistics.pstdev(embed_lat), 3),
                "min": round(min(embed_lat), 3), "max": round(max(embed_lat), 3),
            },
        },
    }

    print(json.dumps(result, indent=2))
    with open(os.path.join(RESULTS_DIR, "recognition_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved results/recognition_results.json")


if __name__ == "__main__":
    main()
