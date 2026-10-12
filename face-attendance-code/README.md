# Reference implementation and experiment code

This folder contains the code and raw results behind the numbers in Tables IX-XIV of
"Offline First Face Attendance - Final.docx". Every number in those tables was
produced by running this code; nothing was hand-typed or estimated.

## What each script does

| Script | Produces |
|---|---|
| `extract_lfw_subset.py` | Selectively pulls a subset of images out of the LFW-funneled archive (90 identities with 2+ images for genuine pairs, 320 single-image identities for impostor pairs) without extracting the full 13,233-image dataset. |
| `recognition_eval.py` | Runs YuNet (face detection) + SFace (MobileFaceNet-backbone embedding, OpenCV Zoo, Apache-2.0) on the LFW subset, builds genuine/impostor pairs, computes accuracy/EER/threshold → **Table XIV**, plus detection/embedding latency rows of **Table XIII**. |
| `matching_latency.py` | Times cosine-similarity matching of a probe embedding against 10 enrolled templates → one row of **Table XIII**. |
| `sync_engine.py` | Real FastAPI + SQLite offline-first synchronization engine (idempotent upsert, capped exponential backoff + jitter, UUIDv4 event IDs) exercised through 6 fault-injection scenarios (duplicate delivery, interrupted sync, retry-with-failures, etc.) → **Tables IX, X, XI**, plus sync-latency rows of **Table XIII**. |
| `liveness_eval.py` | Block-wise Local Binary Pattern histograms + RBF-SVM trained/evaluated on the NUAA Photograph Imposter Database's own official train/test split (3,491 train / 9,123 test images) → **Table XII**, plus the liveness-feature-extraction row of **Table XIII**. |
| `edit_docx.py` | Reads all four `results/*.json` files and writes them into the original .docx's tables and prose, producing the Final.docx. |

## Honesty notes (also stated in the paper itself, Section 8.13 and the Research Integrity Note)

- **Backoff timers in `sync_engine.py` are scaled 1000x** (1 ms / 60 ms instead of
  1 s / 60 s) purely so a 300-event experiment finishes in seconds instead of tens of
  minutes. The retry/backoff *algorithm* (doubling + jitter + cap) is unchanged; only
  wall-clock duration is affected, not any counted outcome.
- **LFW pairs are locally generated**, not the official `pairs.txt` (that host,
  `vis-www.cs.umass.edu`, was unreachable from this environment). Genuine/impostor
  pairs follow the same same-identity/different-identity definition, just not the
  exact official fold split — this is why the reproduced accuracy (~95.6%) differs
  from the officially published SFace number (99.40%) despite using the same model.
- **The liveness classifier is a classical LBP+SVM baseline, not the MobileNetV2
  transfer-learning model** used in the base paper [1]. No camera or Android device
  was available to reproduce their exact pipeline; the NUAA dataset (which the base
  paper also uses) was used with its official split instead.
- **All latency numbers are desktop-CPU timings**, not Android on-device timings.
  True Android latency requires physical hardware.

## Reproducing

Requires Python 3.11 with: `numpy opencv-python-headless fastapi httpx python-docx
scikit-learn scikit-image`. Then, in order:

```
python extract_lfw_subset.py <path-to-lfw-funneled.tgz> <output-dir>
python recognition_eval.py
python matching_latency.py
python sync_engine.py
python liveness_eval.py
python edit_docx.py <path-to-original-docx>
```

Datasets used (both freely downloadable, non-commercial research use):
- LFW-funneled: https://ndownloader.figshare.com/files/5976015 (mirror of Huang et al. 2007)
- NUAA Photograph Imposter Database: https://parnec.nuaa.edu.cn (Tan et al., ECCV 2010) —
  request access per the database's own release terms (non-commercial research only).

Models used (OpenCV Zoo, Apache License 2.0):
- YuNet: `models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- SFace: `models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
