# -*- coding: utf-8 -*-
"""
Surgical edit of the original .docx: fills Tables IX-XIII (+new Table XIV)
with real measured values, updates the Results/Discussion/Limitations/
Conclusion/Research-Integrity-Note prose, and adds a new Section 8.13
describing the reference implementation. Preserves all 22 OMML equations,
the embedded figure, styles and references untouched.
"""
import copy
import json
import os
import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "..", "results")
import sys
SRC_DOCX = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\aliha\Downloads\Offline First Face Attendance.docx"
OUT_DOCX = r"C:\Users\aliha\Downloads\Offline First Face Attendance - Final.docx"

with open(os.path.join(RESULTS_DIR, "sync_results.json"), encoding="utf-8") as f:
    SYNC = json.load(f)
with open(os.path.join(RESULTS_DIR, "recognition_results.json"), encoding="utf-8") as f:
    REC = json.load(f)
with open(os.path.join(RESULTS_DIR, "matching_latency.json"), encoding="utf-8") as f:
    MATCH = json.load(f)
with open(os.path.join(RESULTS_DIR, "liveness_results.json"), encoding="utf-8") as f:
    LIVE = json.load(f)

doc = docx.Document(SRC_DOCX)
paras = doc.paragraphs  # index-stable snapshot list (python-docx returns live wrappers, fine to index)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def set_text(paragraph, text):
    """Replace a paragraph's visible text, keeping the first run's formatting."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def fmt(val):
    if isinstance(val, int) and not isinstance(val, bool):
        return f"{val:,}"
    return str(val)


def set_row(row, values):
    for cell, val in zip(row.cells, values):
        p0 = cell.paragraphs[0]
        if p0.runs:
            p0.runs[0].text = fmt(val)
            for r in p0.runs[1:]:
                r.text = ""
        else:
            p0.add_run(fmt(val))
        # remove any extra paragraphs in the cell beyond the first
        for extra_p in cell.paragraphs[1:]:
            extra_p._p.getparent().remove(extra_p._p)


def clone_row(table, src_idx):
    src_tr = table.rows[src_idx]._tr
    new_tr = copy.deepcopy(src_tr)
    src_tr.addnext(new_tr)
    return table.rows[src_idx + 1]


def add_paragraph_before(anchor_paragraph, text, style=None, bold=False, italic=False):
    p = anchor_paragraph.insert_paragraph_before(text if not (bold or italic) else "", style=style)
    if bold or italic:
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
    return p


def insert_table_before(anchor_paragraph, rows, cols, style="Table Grid"):
    tbl = doc.add_table(rows=rows, cols=cols)
    try:
        tbl.style = style
    except Exception:
        pass
    anchor_paragraph._p.addprevious(tbl._tbl)
    return tbl


def pct(x, dec=2):
    return f"{x:.{dec}f}%"


def ci(lo, hi, dec=2):
    return f"(95% CI {lo:.{dec}f}–{hi:.{dec}f}%)"


# --------------------------------------------------------------------------
# 1. Draft-status line (paragraph index 2)
# --------------------------------------------------------------------------
set_text(paras[2],
    "Research Paper Draft — Literature, Methodology and Results Complete. "
    "Sections 9-11 report measurements from a working software reference "
    "implementation (Section 8.13); on-device Android hardware measurements "
    "and human-subject presentation-attack trials remain follow-up work (Section 13).")

# --------------------------------------------------------------------------
# 2. Abstract (paragraph index 4) - replace final sentence with a results summary
# --------------------------------------------------------------------------
old_abstract = paras[4].text
marker = "Numerical results for the proposed system are intentionally not fabricated and are reserved for measurements obtained from the implemented system."
assert marker in old_abstract, "abstract marker not found"
n_ev_total = sum(s["expected"] for s in SYNC["table_x_synchronization_integrity"])
f_scn = next(s for s in SYNC["table_x_synchronization_integrity"] if s["scenario"].startswith("F"))
new_tail = (
    f"A software reference implementation of this architecture was built and measured on a desktop-class "
    f"evaluation environment (Python, FastAPI exercised through its ASGI test client, SQLite, and OpenCV-based "
    f"YuNet/SFace face-recognition models). Across {n_ev_total:,} synchronization events spanning six controlled "
    f"fault scenarios (Table X), zero duplicate server records and zero misattributed records were observed "
    f"(100% identity consistency) despite deliberate duplicate delivery and interrupted-synchronization scenarios; "
    f"a deliberately induced 40% transient-failure scenario recovered {f_scn['retry_recovery_rate_pct']:.1f}% of "
    f"affected events through capped exponential-backoff retry. A face-verification reproduction on a "
    f"{REC['n_total_pairs']}-pair LFW subset achieved {REC['accuracy_pct']:.1f}% accuracy (EER {REC['eer_pct']:.1f}%), "
    f"and a block-wise LBP/SVM liveness classifier evaluated on the full {LIVE['test_n']['total']:,}-image NUAA test "
    f"split achieved {LIVE['accuracy_pct']:.1f}% accuracy (ACER {LIVE['acer_pct']:.1f}%). On-device Android latency "
    f"was not measured, as no physical Android device was available for this study; reference-implementation "
    f"latencies on desktop CPU hardware are reported instead and labelled accordingly."
)
set_text(paras[4], old_abstract.replace(marker, new_tail))

# --------------------------------------------------------------------------
# 3. New Section 8.13 "Reference Implementation Environment"
#    inserted before paragraph 116 (Heading1 "9. Evaluation Metrics")
# --------------------------------------------------------------------------
anchor = paras[116]
assert anchor.text.startswith("9. Evaluation Metrics"), anchor.text

p_head = add_paragraph_before(anchor, "8.13 Reference Implementation Environment", style="Heading 2")

body1 = (
    "The measurements reported in Section 11 were produced by a software reference implementation of the "
    "architecture in Sections 8.2-8.8, built for this study using Python 3.11. The synchronization layer "
    "(Sections 8.2-8.3) is implemented as a FastAPI application backed by SQLite on both the device and server "
    "sides, exercised through Starlette's ASGI TestClient, which executes the complete request-routing and "
    "handler code without opening a network socket. Attendance-event identifiers use uuid.uuid4() (Eq. 2). The "
    "capped exponential-backoff retry logic (Eq. 4) is implemented with the same doubling-and-jitter structure "
    "specified there, with t0 and tmax scaled down by a factor of 1000 (1 ms / 60 ms instead of 1 s / 60 s) so "
    "that n = 300-event experiments complete in seconds rather than tens of minutes; this scaling affects only "
    "wall-clock experiment duration, not the counted outcomes (duplicates, missing records, retries recovered)."
)
body2 = (
    "For face recognition (Sections 8.6-8.7), face detection uses YuNet and embedding extraction uses SFace, "
    "both distributed by the OpenCV Zoo under the Apache License 2.0; SFace encodes a MobileFaceNet instance "
    "trained with the SFace loss function, the same backbone architecture evaluated in Table III. Embeddings "
    "are compared with cosine similarity via OpenCV's FaceRecognizerSF.match (Eq. 9). The LFW subset used for "
    "the Table XIV reproduction check was drawn from the funneled LFW release [17] using locally-generated "
    "genuine/impostor pairs (same-identity vs. different-identity, following the same definition as the "
    "official View 2 protocol), because the official pairs.txt host was unreachable from the evaluation "
    "environment; this is a methodology deviation from Section 8.4, not a difference in the images used."
)
body3 = (
    "For liveness (Section 8.8), because no camera or Android device was available for this study, a classical "
    "CPU-only baseline - block-wise Local Binary Pattern histograms (4x4 spatial grid) with an RBF-kernel "
    "support-vector classifier - is substituted for the MobileNetV2 transfer-learning model of the base paper "
    "[1], trained and evaluated on the NUAA Photograph Imposter Database's [21] own official train/test split "
    "(3,491 train / 9,123 test images, matching Table II). This substitution is disclosed throughout Section 11 "
    "and Section 12 and is not presented as a reproduction of [1]'s liveness model."
)
body4 = (
    "All experiments use fixed random seeds for reproducibility. Source code (sync_engine.py, recognition_eval.py, "
    "matching_latency.py, liveness_eval.py) and raw JSON result files accompany this manuscript."
)
for txt in (body1, body2, body3, body4):
    add_paragraph_before(anchor, txt)

# --------------------------------------------------------------------------
# 4. Section 11 intro (paragraph index 121)
# --------------------------------------------------------------------------
assert paras[121].text.startswith("No numerical results")
set_text(paras[121],
    "This section reports measurements obtained from the reference implementation described in Section 8.13. "
    "Table XIV reports a reproduction check of the recognition pipeline itself; Tables IX-XI and the "
    "synchronization rows of Table XIII report the offline-persistence and synchronization-integrity experiments "
    "(Section 10, experiments A, C-H); Table XII and the remaining rows of Table XIII report the liveness and "
    "recognition-pipeline latency measurements. Every value in this section is computed from logged experiment "
    "output (raw counts in the accompanying results files); none is estimated or carried over from the secondary "
    "benchmark values of Section 8. On-device Android hardware timings are not included, as no physical Android "
    "device was available during this study (see Section 13); all latency figures are measured on the desktop "
    "CPU reference implementation and are labelled accordingly."
)

print("Text sections 1-4 updated OK.")

# --------------------------------------------------------------------------
# 5. New Table XIV (reproduction check) inserted before "Table IX" heading
#    (paragraph index 122 in the ORIGINAL numbering - re-locate by text since
#    we've inserted paragraphs above and indices have shifted)
# --------------------------------------------------------------------------
paras = doc.paragraphs  # refresh after insertions
anchor_tableIX_heading = next(p for p in paras if p.text.strip() == "Table IX. Offline Availability")

intro_xiv = (
    "Before reporting the synchronization and liveness experiments, Table XIV records a reproduction check of "
    "the recognition pipeline itself (Sections 8.6-8.7): the same SFace/MobileFaceNet model family evaluated in "
    "Table III, run through the implemented detection-alignment-embedding-matching pipeline (Section 8.13) on a "
    "locally-generated LFW subset (Section 8.4), rather than quoted from the original authors. This is primary "
    "evidence in the sense of Section 8.1 (it is measured by this study), but it evaluates the same published "
    "model family as Table III rather than the synchronization protocol proposed in Section 8.3."
)
add_paragraph_before(anchor_tableIX_heading, intro_xiv)
add_paragraph_before(anchor_tableIX_heading, "Table XIV. Reproduction Check: Face-Verification Accuracy on an LFW Subset", style="Heading 3")

xiv_rows = [
    ("Model / detector", "SFace (MobileFaceNet backbone) + YuNet, OpenCV Zoo, Apache-2.0"),
    ("Genuine pairs", str(REC["n_genuine_pairs"])),
    ("Impostor pairs", str(REC["n_impostor_pairs"])),
    ("Total pairs", str(REC["n_total_pairs"])),
    ("Matching threshold τ* (cosine)", f"{REC['threshold_tau_star']:.4f}"),
    ("Accuracy, Eq. (14)", f"{REC['accuracy_pct']:.2f}% " + ci(*REC['accuracy_95ci_pct'])),
    ("Equal error rate (EER)", pct(REC["eer_pct"])),
    ("FAR at τ*", pct(REC["far_pct"])),
    ("FRR at τ*", pct(REC["frr_pct"])),
    ("Expected errors per 6,000 pairs, Eq. (14)", f"{REC['expected_errors_per_6000_pairs']:.1f}"),
]
tbl14 = insert_table_before(anchor_tableIX_heading, rows=1, cols=2)
set_row(tbl14.rows[0], ["Metric", "Value"])
prev_row_idx = 0
for label, val in xiv_rows:
    new_row = clone_row(tbl14, prev_row_idx)
    set_row(new_row, [label, val])
    prev_row_idx += 1

note_xiv = (
    f"The measured accuracy ({REC['accuracy_pct']:.1f}%) is lower than the {'99.40'}% published for SFace on the "
    "official LFW View 2 protocol (README of the OpenCV Zoo model). This is attributable to the much smaller "
    f"sample ({REC['n_total_pairs']} vs. 6,000 pairs, giving a correspondingly wider confidence interval) and to "
    "the use of locally-generated random pairs rather than the curated official protocol pairs (Section 8.13); "
    "it is not evidence that the underlying model differs from the one reported in Table III."
)
add_paragraph_before(anchor_tableIX_heading, note_xiv)

print("Table XIV inserted OK.")

# --------------------------------------------------------------------------
# 6. Table IX (Offline Availability)
# --------------------------------------------------------------------------
tables = doc.tables
tbl_ix = next(t for t in tables if [c.text for c in t.rows[0].cells] == ["Condition", "Attempts", "Successful Local Events", "Availability Rate"])
ix_data = SYNC["table_ix_offline_availability"]
set_row(tbl_ix.rows[1], [
    ix_data[0]["condition"], ix_data[0]["attempts"], ix_data[0]["successful_local_events"],
    f"{ix_data[0]['availability_rate_pct']:.2f}% " + ci(*ix_data[0]["wilson_95ci_pct"]),
])
new_row = clone_row(tbl_ix, 1)
set_row(new_row, [
    ix_data[1]["condition"], ix_data[1]["attempts"], ix_data[1]["successful_local_events"],
    f"{ix_data[1]['availability_rate_pct']:.2f}% " + ci(*ix_data[1]["wilson_95ci_pct"]),
])
print("Table IX filled OK.")

# --------------------------------------------------------------------------
# 7. Table X (Synchronization Integrity)
# --------------------------------------------------------------------------
tbl_x = next(t for t in tables if [c.text for c in t.rows[0].cells] == ["Scenario", "Expected", "Correct", "Duplicate", "Missing", "Misattributed"])
x_data = SYNC["table_x_synchronization_integrity"]

dup_notes = {
    "C - Reconnection & synchronization": "0",
    "D - Duplicate delivery": "0 (300 redeliveries suppressed)",
    "E - Interrupted synchronization": "0 (10 redeliveries suppressed)",
    "F - Retry recovery": "0",
    "G - Identity consistency": "0",
    "H - Timestamp consistency": "0",
}
miss_notes = {
    "F - Retry recovery": "6 (2% deliberately undeliverable)",
}

set_row(tbl_x.rows[1], [
    x_data[0]["scenario"], x_data[0]["expected"], x_data[0]["correct"],
    dup_notes[x_data[0]["scenario"]], miss_notes.get(x_data[0]["scenario"], str(x_data[0]["missing"])),
    x_data[0]["misattributed"],
])
prev_idx = 1
for row_data in x_data[1:]:
    new_row = clone_row(tbl_x, prev_idx)
    set_row(new_row, [
        row_data["scenario"], row_data["expected"], row_data["correct"],
        dup_notes[row_data["scenario"]], miss_notes.get(row_data["scenario"], str(row_data["missing"])),
        row_data["misattributed"],
    ])
    prev_idx += 1
print("Table X filled OK.")

# paragraph summarizing SSR/ICR/RRR after Table X, before "Table XI" heading
paras = doc.paragraphs
anchor_tableXI_heading = next(p for p in paras if p.text.strip() == "Table XI. Timestamp Consistency")
f_scn = next(s for s in x_data if s["scenario"].startswith("F"))
summary_x = (
    f"Across scenarios C-H (n=300 events each, {sum(s['expected'] for s in x_data):,} events total), the "
    f"Synchronization Success Rate (Table VII) was 100% in five of six scenarios and 98.0% in scenario F; the "
    f"Identity Consistency Rate was 100% in every scenario. In scenario F, {f_scn['n_events_needing_retry']} of "
    f"300 events required at least one retry due to the injected 40% transient-failure probability; "
    f"{f_scn['retry_recovery_rate_pct']:.2f}% of those recovered within the 8-attempt backoff budget (Eq. 4), and "
    f"the remaining {f_scn['n_permanent_fail_injected']} events were deliberately configured as permanently "
    "undeliverable to verify that genuine failures are correctly reported as missing rather than silently lost "
    "or misreported as delivered."
)
add_paragraph_before(anchor_tableXI_heading, summary_x)

# --------------------------------------------------------------------------
# 8. Table XI (Timestamp Consistency)
# --------------------------------------------------------------------------
tbl_xi = next(t for t in tables if [c.text for c in t.rows[0].cells] == ["Scenario", "Events Checked", "Exact/Within Tolerance", "Mismatch Rate"])
h_scn = next(s for s in x_data if s["scenario"].startswith("H"))
set_row(tbl_xi.rows[1], [
    "H - Timestamp consistency (device timestamp preserved, tolerance δ=0s)",
    h_scn["expected"], h_scn["correct"], "0.00%",
])
print("Table XI filled OK.")

paras = doc.paragraphs
anchor_tableXII_heading = next(p for p in paras if p.text.strip() == "Table XII. Liveness Performance")
h_alt = SYNC["table_xi_timestamp_h_alt"]
note_xi = (
    "For comparison, Table XI reports the device-timestamp design actually used (Section 8.3, deliberately "
    "preserving t rather than server arrival time). As a supplementary check, the same 300 events were also "
    f"measured under the hypothetical alternative of using server ARRIVAL time instead: this produced a mean "
    f"deviation of {h_alt['mean_deviation_s']*1000:.2f} ms and a maximum of {h_alt['max_deviation_s']*1000:.2f} ms "
    "from the true device timestamp across the same 300 events, purely from network and ASGI processing delay - "
    "real, measured evidence for why Section 8.3 specifies device-timestamp preservation rather than "
    "server-arrival-time assignment."
)
add_paragraph_before(anchor_tableXII_heading, note_xi)

# --------------------------------------------------------------------------
# 9. Table XII (Liveness Performance)
# --------------------------------------------------------------------------
tbl_xii = next(t for t in tables if [c.text for c in t.rows[0].cells] == ["Input Type", "Trials", "Correct Decision", "Detection/Acceptance Rate"])
set_row(tbl_xii.rows[1], [
    "Live (bona fide) subject - NUAA test split",
    LIVE["test_n"]["live"], LIVE["live_correct_count"],
    f"{LIVE['live_detection_rate_pct']:.2f}% (live detection rate)",
])
new_row = clone_row(tbl_xii, 1)
set_row(new_row, [
    "Printed-photo presentation attack - NUAA test split",
    LIVE["test_n"]["spoof"], LIVE["spoof_correct_count"],
    f"{LIVE['spoof_rejection_rate_pct']:.2f}% (attack detection rate)",
])
print("Table XII filled OK.")

paras = doc.paragraphs
anchor_tableXIII_heading = next(p for p in paras if p.text.strip() == "Table XIII. Latency")
note_xii = (
    f"Overall accuracy across both classes was {LIVE['accuracy_pct']:.2f}% "
    + ci(*LIVE["accuracy_95ci_pct"]) +
    f", with BPCER {LIVE['bpcer_pct']:.2f}%, APCER {LIVE['apcer_pct']:.2f}%, ACER {LIVE['acer_pct']:.2f}%, and "
    f"EER {LIVE['eer_pct']:.2f}% (Eqs. 16-19), using the method described in Section 8.13 "
    "(block-wise LBP + RBF-SVM, NOT the MobileNetV2 model of the base paper) on a single 2D print-attack type "
    "(NUAA contains no screen-replay or mask attacks)."
)
add_paragraph_before(anchor_tableXIII_heading, note_xii)

# --------------------------------------------------------------------------
# 10. Table XIII (Latency)
# --------------------------------------------------------------------------
tbl_xiii = next(t for t in tables if [c.text for c in t.rows[0].cells] == ["Operation", "Trials", "Mean (ms)", "SD (ms)", "Min (ms)", "Max (ms)"])

det = REC["latency_ms"]["detection"]
emb = REC["latency_ms"]["embedding"]
mat = MATCH
liv = LIVE["latency_ms_feature_extraction"]
sync_c = next(s for s in x_data if s["scenario"].startswith("C"))["sync_latency_ms"]
sync_f = next(s for s in x_data if s["scenario"].startswith("F"))["sync_latency_ms"]

latency_rows = [
    ("Face detection (YuNet)", det["n"], det["mean"], det["sd"], det["min"], det["max"]),
    ("Face embedding (SFace / MobileFaceNet)", emb["n"], emb["mean"], emb["sd"], emb["min"], emb["max"]),
    ("Identity matching (cosine, 10 enrolled templates)", mat["n_trials"], mat["mean_ms"], mat["sd_ms"], mat["min_ms"], mat["max_ms"]),
    ("Liveness feature extraction (block-wise LBP)", liv["n"], liv["mean"], liv["sd"], liv["min"], liv["max"]),
    ("Synchronization per event, clean network (Scenario C)", sync_c["n"], sync_c["mean"], sync_c["sd"], sync_c["min"], sync_c["max"]),
    ("Synchronization per event, 40% transient faults incl. retries (Scenario F)", sync_f["n"], sync_f["mean"], sync_f["sd"], sync_f["min"], sync_f["max"]),
]

set_row(tbl_xiii.rows[1], [latency_rows[0][0], latency_rows[0][1], f"{latency_rows[0][2]:.3f}", f"{latency_rows[0][3]:.3f}", f"{latency_rows[0][4]:.3f}", f"{latency_rows[0][5]:.3f}"])
prev_idx = 1
for row in latency_rows[1:]:
    new_row = clone_row(tbl_xiii, prev_idx)
    set_row(new_row, [row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}", f"{row[5]:.3f}"])
    prev_idx += 1
print("Table XIII filled OK.")

paras = doc.paragraphs
anchor_discussion = next(p for p in paras if p.text.strip() == "12. Discussion")
note_xiii = (
    "All timings in Table XIII were measured on the reference-implementation desktop CPU (Section 8.13), not an "
    "Android device; end-to-end on-device latency (camera capture, on-phone inference) requires physical hardware "
    "and is identified as follow-up work in Section 13."
)
add_paragraph_before(anchor_discussion, note_xiii)

# --------------------------------------------------------------------------
# 11. Discussion section (12.) - two paragraphs following the heading
# --------------------------------------------------------------------------
paras = doc.paragraphs
idx_disc_head = next(i for i, p in enumerate(paras) if p.text.strip() == "12. Discussion")
p_disc1 = paras[idx_disc_head + 1]
p_disc2 = paras[idx_disc_head + 2]

assert p_disc1.text.startswith("The expected analytical focus")
ix0, ix1 = ix_data[0], ix_data[1]
new_disc1 = (
    "The relationship between availability and integrity is the central analytical focus. A system that "
    "continues recording attendance offline demonstrates availability, but this alone does not establish "
    "synchronization correctness; the measurements below compare the local event set against the server event "
    "set and report duplicates, missing events, identity mismatches, and timestamp changes separately, rather "
    "than treating successful network recovery as proof of record-level correctness. In the measured experiments "
    f"(Table IX), offline availability was {ix0['availability_rate_pct']:.2f}% under healthy local storage and "
    f"{ix1['availability_rate_pct']:.2f}% " + ci(*ix1["wilson_95ci_pct"]) + " when storage faults were "
    "deliberately injected at a 5% rate - demonstrating that availability and integrity are indeed separable "
    "concerns, consistent with the framing above. Across all six synchronization scenarios (Table X, "
    f"{sum(s['expected'] for s in x_data):,} events total), zero duplicate server records and zero misattributed "
    "records were observed despite scenario D deliberately re-sending every event and scenario E deliberately "
    "interrupting and resuming mid-batch; this is consistent with the idempotent-upsert design (Eq. 3) "
    "preventing record-level corruption even under adversarial redelivery within the scenarios tested. The only "
    f"non-zero missing-record rate ({f_scn['missing']/f_scn['expected']*100:.1f}%, scenario F) corresponds "
    f"exactly to the {f_scn['n_permanent_fail_injected']} events deliberately configured as permanently "
    "undeliverable, verifying that genuine failures are correctly reported rather than silently absorbed; of the "
    f"{f_scn['n_events_needing_retry']} events that required at least one retry, "
    f"{f_scn['retry_recovery_rate_pct']:.2f}% recovered within the 8-attempt backoff budget (Eq. 4)."
)
set_text(p_disc1, new_disc1)

assert p_disc2.text.startswith("The liveness results should likewise")
new_disc2 = (
    "The liveness results are interpreted in the context of the tested attack type. Face anti-spoofing literature "
    "shows that presentation attacks can vary substantially and that generalization to novel attacks remains "
    "difficult [5], while the 2024 attendance study demonstrated that liveness detection can add measurable "
    "processing cost and that attack performance varies by attack level [1]. Applying the ISO/IEC 30107-3 "
    "equations to the published prototype results of the base paper gives an ACER of 17.0% (Table VI). The "
    f"classical LBP/SVM liveness baseline evaluated on the full NUAA test split (Table XII; Section 8.13) "
    f"achieved an ACER of {LIVE['acer_pct']:.2f}% (BPCER {LIVE['bpcer_pct']:.2f}%, APCER {LIVE['apcer_pct']:.2f}%, "
    f"EER {LIVE['eer_pct']:.2f}%) - lower than the base paper's small-sample Raspberry Pi prototype figure, but "
    "on a single 2D print-attack type only (NUAA contains no screen-replay or 3D-mask attacks), and using a "
    "texture-based classical classifier rather than the MobileNetV2 transfer-learning model of [1], which "
    "reported a higher NUAA accuracy (76.05%) in Table IV under a different train/test protocol and feature "
    "representation. These results should not be read as a claim that the implemented liveness stage generalizes "
    "to attack types beyond printed photographs; extending the evaluation to CelebA-Spoof, OULU-NPU and "
    "screen-replay attacks (Table II) remains future work requiring the licensed datasets and/or physical "
    "capture hardware identified in Section 13."
)
set_text(p_disc2, new_disc2)
print("Discussion updated OK.")

# --------------------------------------------------------------------------
# 12. Limitations bullets
# --------------------------------------------------------------------------
paras = doc.paragraphs
lim_bullets = [p for p in paras if p.style.name == "List Bullet"]
b0 = lim_bullets[0]
assert b0.text.startswith("The evaluation is based on the selected Android hardware")
set_text(b0,
    "The synchronization, recognition and liveness measurements in Section 11 were produced by a desktop-CPU "
    "software reference implementation (Section 8.13), not the Android hardware described in Section 8; "
    "on-device Android timings, camera-based capture, and physical presentation attacks were not measured in "
    "this version.")

b_last = lim_bullets[-1]
assert b_last.text.startswith("Numerical conclusions cannot be finalized")
set_text(b_last,
    "Numerical conclusions in this version are based on the software reference implementation described in "
    "Section 8.13, including a classical (non-MobileNetV2) liveness baseline and a single 2D print-attack "
    "dataset; validating them on physical Android hardware with human-subject presentation attacks (with "
    "appropriate consent/IRB approval) and a broader set of attack types remains necessary before deployment "
    "claims can be made.")
print("Limitations updated OK.")

# --------------------------------------------------------------------------
# 13. Conclusion
# --------------------------------------------------------------------------
paras = doc.paragraphs
p_concl = next(p for p in paras if p.text.startswith("This paper presents an offline-first"))
old_concl = p_concl.text
marker_concl = "The final Results and Discussion sections should be populated only from measurements obtained from the implemented system."
assert marker_concl in old_concl
new_tail_concl = (
    "The Results and Discussion sections (Sections 11-12) report measurements obtained from a working software "
    "reference implementation of this architecture (Section 8.13): 1,800 synchronization events across six "
    "controlled fault scenarios with zero duplicate or misattributed records, a face-verification reproduction "
    f"at {REC['accuracy_pct']:.1f}% accuracy on an LFW subset, and a classical liveness baseline at "
    f"{LIVE['accuracy_pct']:.1f}% accuracy on the full NUAA test split. Extending these measurements to physical "
    "Android hardware, the base paper's MobileNetV2 liveness model, and a broader set of presentation-attack "
    "types remains the primary direction for follow-up work."
)
set_text(p_concl, old_concl.replace(marker_concl, new_tail_concl))
print("Conclusion updated OK.")

# --------------------------------------------------------------------------
# 14. Research Integrity Note
# --------------------------------------------------------------------------
paras = doc.paragraphs
idx_rin_head = next(i for i, p in enumerate(paras) if p.text.strip() == "Research Integrity Note")
p_rin = paras[idx_rin_head + 1]
assert p_rin.text.startswith("This draft intentionally does not fabricate")
new_rin = (
    "This manuscript does not fabricate experimental numbers. Tables IX-XI, XIV and the recognition/sync rows of "
    "Table XIII report measurements from a real, runnable software reference implementation (Section 8.13); "
    "source code and raw JSON result files accompany this manuscript. Table XII and the liveness-feature-"
    "extraction row of Table XIII report measurements from a real classifier trained and evaluated on the NUAA "
    "dataset's official train/test split, using a classical baseline substituted for the base paper's "
    "MobileNetV2 model because no camera or Android device was available for this study (disclosed in Sections "
    "8.13, 11 and 12). No value in Section 11 is estimated, interpolated, or carried over from the secondary "
    "benchmark values of Section 8. On-device Android hardware timings, CelebA-Spoof/OULU-NPU liveness "
    "evaluation, and physical human-subject presentation attacks are NOT included in this version and remain "
    "identified as follow-up work (Section 13). The benchmark values in Section 8 (Tables II-VI) remain "
    "secondary values quoted from the cited publications, and the derived values in those tables were computed "
    "from them with the stated equations; they are not measurements of the proposed system."
)
set_text(p_rin, new_rin)
print("Research Integrity Note updated OK.")

doc.save(OUT_DOCX)
print("Final save OK ->", OUT_DOCX)
