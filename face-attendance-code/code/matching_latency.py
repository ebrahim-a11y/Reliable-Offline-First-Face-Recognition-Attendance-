"""Measure real matching latency: probe embedding vs. 10 enrolled templates
(cosine similarity, Eq. 9), using genuine SFace embeddings from the LFW
subset already extracted."""
import cv2
import os
import json
import time
import statistics
import glob

BASE = os.path.dirname(__file__)
SUBSET_DIR = os.path.join(BASE, "..", "data", "lfw_subset", "lfw_funneled")
MODELS_DIR = os.path.join(BASE, "..", "models")
RESULTS_DIR = os.path.join(BASE, "..", "results")

detector = cv2.FaceDetectorYN.create(
    os.path.join(MODELS_DIR, "yunet.onnx"), "", (320, 320),
    score_threshold=0.7, nms_threshold=0.3, top_k=5000,
)
recognizer = cv2.FaceRecognizerSF.create(os.path.join(MODELS_DIR, "sface.onnx"), "")


def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    aligned = recognizer.alignCrop(img, faces[0])
    return recognizer.feature(aligned)


def main():
    people = sorted(glob.glob(os.path.join(SUBSET_DIR, "*")))[:15]
    templates = []
    for p in people:
        imgs = glob.glob(os.path.join(p, "*.jpg"))
        if not imgs:
            continue
        emb = get_embedding(imgs[0])
        if emb is not None:
            templates.append(emb)
        if len(templates) >= 10:
            break

    probe_paths = []
    for p in people:
        probe_paths.extend(glob.glob(os.path.join(p, "*.jpg")))
    probe_paths = probe_paths[:200] if len(probe_paths) >= 200 else (probe_paths * 20)[:200]

    latencies = []
    for pth in probe_paths:
        probe = get_embedding(pth)
        if probe is None:
            continue
        t0 = time.perf_counter()
        best = max(float(recognizer.match(probe, t, cv2.FaceRecognizerSF_FR_COSINE)) for t in templates)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    result = {
        "n_templates": len(templates),
        "n_trials": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 4),
        "sd_ms": round(statistics.pstdev(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
    }
    print(json.dumps(result, indent=2))
    with open(os.path.join(RESULTS_DIR, "matching_latency.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
