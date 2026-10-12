"""
Selectively extract a subset of the LFW-funneled dataset for the recognition
reproduction experiment, without extracting all 13,233 images.

Selection rule (disclosed in the paper as a methodology note):
  - Identities with >= 2 images are candidates for GENUINE pairs.
  - Identities with exactly 1 image are candidates for IMPOSTOR pairs.
  - Selection uses a fixed random seed for reproducibility.
"""
import tarfile
import random
import os
import json
import sys

TGZ_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/lfw-funneled.tgz"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "../data/lfw_subset"
SEED = 42
N_GENUINE_IDENTITIES = 90     # identities with >=2 images to draw genuine pairs from
N_IMPOSTOR_SINGLES = 320      # single-image identities to draw impostor pairs from

random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Scanning {TGZ_PATH} for member list (no extraction yet)...")
with tarfile.open(TGZ_PATH, "r:gz") as tf:
    members = tf.getnames()
    # members look like: lfw_funneled/George_HW_Bush/George_HW_Bush_0001.jpg
    by_identity = {}
    for m in members:
        parts = m.split("/")
        if len(parts) == 3 and parts[2].lower().endswith(".jpg"):
            identity = parts[1]
            by_identity.setdefault(identity, []).append(m)

    identities = list(by_identity.keys())
    multi = [i for i in identities if len(by_identity[i]) >= 2]
    single = [i for i in identities if len(by_identity[i]) == 1]
    print(f"Total identities: {len(identities)}  multi-image: {len(multi)}  single-image: {len(single)}")

    random.shuffle(multi)
    random.shuffle(single)

    chosen_multi = multi[:N_GENUINE_IDENTITIES]
    chosen_single = single[:N_IMPOSTOR_SINGLES]

    to_extract = []
    manifest = {"genuine_identities": {}, "impostor_identities": {}}
    for ident in chosen_multi:
        imgs = by_identity[ident][:4]  # cap at 4 images per identity
        to_extract.extend(imgs)
        manifest["genuine_identities"][ident] = [os.path.basename(p) for p in imgs]
    for ident in chosen_single:
        imgs = by_identity[ident]
        to_extract.extend(imgs)
        manifest["impostor_identities"][ident] = [os.path.basename(p) for p in imgs]

    print(f"Extracting {len(to_extract)} images to {OUT_DIR} ...")
    for member_name in to_extract:
        member = tf.getmember(member_name)
        tf.extract(member, path=OUT_DIR)

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("Done.")
print(f"Genuine-pair identities: {len(manifest['genuine_identities'])}")
print(f"Impostor-pair identities: {len(manifest['impostor_identities'])}")
