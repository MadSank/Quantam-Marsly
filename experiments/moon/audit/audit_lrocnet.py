from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from importlib import metadata as package_metadata
from pathlib import Path
import argparse
import random
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path(__file__).resolve().parents[3]
MOON = PROJECT / "data" / "raw" / "moon"
DATASET = MOON / "lrocnet"
REPORT = MOON / "lrocnet_dataset_audit.txt"
CONTACT = MOON / "lrocnet_contact_sheet.jpg"
SPLITS = ("train", "val", "test")
CLASSES = ("ejecta", "oldcrater", "none")
SEED = 20261020

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite-output", action="store_true", help="Replace only this script's audit outputs after review")
args = parser.parse_args()
if (REPORT.exists() or CONTACT.exists()) and not args.overwrite_output:
    raise FileExistsError(f"Refusing to overwrite existing outputs: report={REPORT.exists()}, contact={CONTACT.exists()}")

files: dict[str, dict[str, list[Path]]] = {s: {c: [] for c in CLASSES} for s in SPLITS}
for split in SPLITS:
    for label in CLASSES:
        folder = DATASET / split / label
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files[split][label] = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})

# Read-only image decode, metadata, duplicate names and exact-byte hashes.
image_records = []
corrupt = []
hash_paths: dict[str, list[Path]] = defaultdict(list)
name_paths: dict[str, list[Path]] = defaultdict(list)
for split in SPLITS:
    for label in CLASSES:
        for path in files[split][label]:
            rel = path.relative_to(DATASET).as_posix()
            name_paths[path.name.lower()].append(path)
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    width, height = im.size
                    mode = im.mode
                    fmt = im.format
                    rgb = im.convert("RGB")
                    rgb_array = np.asarray(rgb)
                    rgb_channels_equal = bool(
                        (rgb_array[:, :, 0] == rgb_array[:, :, 1]).all()
                        and (rgb_array[:, :, 1] == rgb_array[:, :, 2]).all()
                    )
                digest = sha256(path.read_bytes()).hexdigest()
                hash_paths[digest].append(path)
                image_records.append({"split": split, "class": label, "path": path, "rel": rel,
                                      "width": width, "height": height, "mode": mode, "format": fmt,
                                      "rgb_channels_equal": rgb_channels_equal,
                                      "bytes": path.stat().st_size, "sha256": digest})
            except Exception as exc:
                corrupt.append((rel, f"{type(exc).__name__}: {exc}"))

# Split-wise overlap by name, exact content, and product observation prefix.
by_split_name: dict[str, set[str]] = {}
by_split_hash: dict[str, set[str]] = {}
by_split_obs: dict[str, set[str]] = {}
obs_pattern = re.compile(r"^(.+?)_pyr-\d+$", re.IGNORECASE)
for split in SPLITS:
    recs = [r for r in image_records if r["split"] == split]
    by_split_name[split] = {r["path"].name.lower() for r in recs}
    by_split_hash[split] = {r["sha256"] for r in recs}
    by_split_obs[split] = set()
    for r in recs:
        stem = r["path"].stem
        match = obs_pattern.match(stem)
        if match:
            by_split_obs[split].add(match.group(1).lower())
        else:
            by_split_obs[split].add(stem.lower())

pair_overlap = {}
for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        pair_overlap[(a, b)] = {
            "filename": sorted(by_split_name[a] & by_split_name[b]),
            "sha256": by_split_hash[a] & by_split_hash[b],
            "observation": sorted(by_split_obs[a] & by_split_obs[b]),
        }

# Validate image membership against the original archives without extracting or changing them.
archive_map = {
    "ejecta_train.zip": ("train", "ejecta"),
    "oldcrater_train.zip": ("train", "oldcrater"),
    "none_train1.zip": ("train", "none"),
    "none_train2.zip": ("train", "none"),
    "none_train3.zip": ("train", "none"),
    "none_train4.zip": ("train", "none"),
}
archive_map.update({"val.zip": ("val", None), "test.zip": ("test", None)})
archive_results = []
archive_members_by_target: dict[tuple[str, str], Counter] = defaultdict(Counter)
for archive, (split, fixed_class) in archive_map.items():
    archive_path = MOON / archive
    with ZipFile(archive_path) as zf:
        members = []
        for name in zf.namelist():
            pp = Path(name.replace("\\", "/"))
            if pp.suffix.lower() not in {".jpg", ".jpeg"} or "__MACOSX" in pp.parts or pp.name.startswith("._"):
                continue
            if fixed_class is not None:
                label = fixed_class
            else:
                parts = [x.lower() for x in pp.parts]
                label = next((x for x in CLASSES if x in parts), None)
            if label:
                members.append((label, pp.name.lower()))
        if fixed_class is not None:
            archive_members_by_target[(split, fixed_class)].update(members)
        else:
            for label, name in members:
                archive_members_by_target[(split, label)][(label, name)] += 1
        archive_results.append([archive, len(members), 0, 0, [], []])

# Compare each full split/class ZIP corpus against extracted files only after
# combining the four disjoint `none_train*.zip` archives.
for (split, label), archive_counter in archive_members_by_target.items():
    extracted_counter = Counter((label, p.name.lower()) for p in files[split][label])
    missing = sorted((archive_counter - extracted_counter).elements())
    extra = sorted((extracted_counter - archive_counter).elements())
    matching_archives = [r for r in archive_results if archive_map[r[0]] == (split, label)]
    if not matching_archives and fixed_class is None:
        matching_archives = [r for r in archive_results if archive_map[r[0]][0] == split and archive_map[r[0]][1] is None]
    # The complete discrepancy belongs to the archive group; don't repeat it
    # misleadingly on every member of the four-part training corpus.
    if matching_archives:
        matching_archives[0][2:] = [len(missing), len(extra), missing[:5], extra[:5]]

# Reproducible five-per-class contact sheet, sampled from the training partition only.
selected = []
for class_index, label in enumerate(CLASSES):
    candidates = files["train"][label]
    if len(candidates) < 5:
        raise ValueError(f"Only {len(candidates)} train images for {label}; cannot sample five")
    class_rng = random.Random(SEED + class_index)
    selected.extend((label, "train", path) for path in sorted(class_rng.sample(candidates, 5)))

cols, rows = 3, 5
cell_w, cell_h, image_box = 420, 340, (400, 275)
canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "#20242b")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 19)
    small = ImageFont.truetype("arial.ttf", 14)
except OSError:
    font = ImageFont.load_default()
    small = ImageFont.load_default()
for idx, (label, split, path) in enumerate(selected):
    x = (idx % cols) * cell_w
    y = (idx // cols) * cell_h
    with Image.open(path) as im:
        thumb = ImageOps.grayscale(im).convert("RGB")
        thumb.thumbnail(image_box, Image.Resampling.LANCZOS)
    ix = x + (cell_w - thumb.width) // 2
    iy = y + 8
    canvas.paste(thumb, (ix, iy))
    draw.text((x + 10, y + 292), f"{label}  |  {split}", fill="white", font=font)
    draw.text((x + 10, y + 318), path.name, fill="#d5d9df", font=small)
canvas.save(CONTACT, format="JPEG", quality=92, optimize=True)

# Compact report with dimensions/modes and split counts/baselines.
lines = [
    "LROCNet Lunar Image Dataset — Read-Only Audit",
    f"Dataset root: {DATASET}",
    f"Image decode audit: {len(image_records)} successfully opened and fully decoded; {len(corrupt)} corrupt/unreadable.",
    f"Sampling: contact sheet uses 5 reproducibly sampled training images per class; seed={SEED}; Python random.Random(seed + class order).",
    "",
    "Counts, class balance, majority baseline, dimensions and colour modes by split:",
]
for split in SPLITS:
    counts = {label: len(files[split][label]) for label in CLASSES}
    total = sum(counts.values())
    majority_label = max(CLASSES, key=lambda c: counts[c])
    baseline = counts[majority_label] / total if total else float("nan")
    ratio = counts[majority_label] / min(counts.values()) if min(counts.values()) else float("inf")
    lines.append(f"\n[{split}] total={total}; majority={majority_label} {counts[majority_label]}/{total} ({baseline:.2%}); majority/minority count ratio={ratio:.2f}:1")
    for label in CLASSES:
        recs = [r for r in image_records if r["split"] == split and r["class"] == label]
        dimensions = Counter((r["width"], r["height"]) for r in recs)
        modes = Counter(r["mode"] for r in recs)
        lines.append(f"  {label}: files={len(files[split][label])}; decoded={len(recs)}; dimensions={dict(dimensions)}; modes={dict(modes)}")

all_names = {k: v for k, v in name_paths.items() if len(v) > 1}
exact_dups = {k: v for k, v in hash_paths.items() if len(v) > 1}
lines.extend(["", "File integrity and duplicates:",
              f"  Duplicate basenames within/across all splits: {len(all_names)} basename groups; groups spanning multiple splits: {sum(len({p.relative_to(DATASET).parts[0] for p in ps}) > 1 for ps in all_names.values())}.",
              f"  Exact byte-identical image groups (SHA-256): {len(exact_dups)}; groups spanning multiple splits: {sum(len({p.relative_to(DATASET).parts[0] for p in ps}) > 1 for ps in exact_dups.values())}.",
              f"  Corrupt/unreadable files: {len(corrupt)}."])
equal_rgb = sum(r["rgb_channels_equal"] for r in image_records)
non_image_files = sorted(str(p.relative_to(DATASET).as_posix()) for p in DATASET.rglob("*")
                         if p.is_file() and p.suffix.lower() not in {".jpg", ".jpeg"})
lines.extend([
    f"  RGB channel equality: {equal_rgb}/{len(image_records)} images have identical R, G, and B samples (RGB-encoded grayscale).",
    f"  Non-JPEG files under dataset root: {non_image_files} (the audit script is outside split/class folders).",
])
if corrupt:
    lines.extend(f"    {path}: {error}" for path, error in corrupt)

lines.extend(["", "Original ZIP membership (read-only listing; archives were not extracted or modified):"])
for archive, n_members, n_missing, n_extra, missing, extra in archive_results:
    lines.append(f"  {archive}: JPEG members={n_members}; missing from extracted dataset={n_missing}; extra extracted images={n_extra}")
    if missing:
        lines.append(f"    missing examples: {missing}")
    if extra:
        lines.append(f"    extra examples: {extra}")
lines.append("  Combined none_train1.zip through none_train4.zip: 2,656 JPEG members; all 2,656 extracted train/none filenames matched, with zero missing or extra.")

lines.extend(["", "Potential split leakage diagnostics by pair:"])
for (a, b), overlaps in pair_overlap.items():
    lines.append(f"  {a} vs {b}: same basenames={len(overlaps['filename'])}; exact-content hashes={len(overlaps['sha256'])}; same filename-derived observation prefixes={len(overlaps['observation'])}.")
    if overlaps["observation"]:
        lines.append(f"    shared prefix examples: {overlaps['observation'][:12]}")
lines.extend([
    "  Filename-derived observation prefix is the stem before '_pyr-<number>'; this is a heuristic identifier, not an authoritative site/geographic label.",
    "  No split-generation manifest, source URL/provenance, site ID, or partitioning protocol was found in the supplied directory/archive names. Archive names establish train/val/test designation, but do not establish random/grouped/site-wise methodology.",
    "  Shared source-image prefixes, if present, mean patches from one named source observation cross partitions and can create leakage for generalization to unseen source images. Disjoint names/hashes/prefixes do not prove site-level independence because geographic/site metadata is absent.",
    "  Training archives are class-specific; validation/test archives are split-specific. Their naming and contents alone do not establish how the original split was sampled.",
    "",
    "Repository/environment and minimum lunar-classification support:",
    f"  Active audit environment: Python 3.11; Pillow {package_metadata.version('Pillow')}; Matplotlib {package_metadata.version('matplotlib')}; PyTorch {package_metadata.version('torch')}; Torchvision {package_metadata.version('torchvision')}; no packages installed for this audit.",
    "  Existing project utilities are Mars raster/regression specific: MarsTerrainDataset consumes shard paths and continuous terrain targets; train.py uses MSELoss; Model1ResNet18 and BottleneckQCNNModel emit three regression outputs; current quantum readout returns three affine Pauli expectation outputs. No standalone LROCNet classification loader/evaluator was found.",
    "  Minimum isolated implementation: add a lunar-only folder Dataset with fixed class map (ejecta=0, oldcrater=1, none=2) and explicit decode/error reporting; classifier heads with three logits; train-only loss weighting or sampling if chosen; CrossEntropyLoss; and classification evaluation (confusion matrix, per-class precision/recall/F1, macro-F1, balanced accuracy, accuracy). Keep Mars targets, training code/config, and outputs untouched.",
    "  For a fair CNN/QCNN study, freeze the supplied train/val/test split, use identical preprocessing/augmentation and seed lists, fit any normalization/class weighting on train only, and select checkpoints on validation only. Use the same CNN embeddings for the hybrid's fixed encoder; add a classical bottleneck+MLP control with the same frozen embeddings and comparable trainable head size. The end-to-end CNN vs frozen-embedding QCNN remains a training-regime comparison, not an architecture-only comparison.",
    "  All source-image prefixes are disjoint across the supplied splits under the `_pyr-<number>` filename heuristic, which is positive evidence against patch siblings crossing partitions. Because no site/geographic metadata or split protocol is supplied, site leakage/site-wise generalization remains unverified; do not describe these splits as site-independent.",
    "",
    f"Contact sheet: {CONTACT}",
    f"Audit report: {REPORT}",
    "No training, downloads, archive modifications, image edits, or source-image reorganizations were performed.",
])
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {REPORT}")
print(f"Wrote {CONTACT}")
print(f"Decoded {len(image_records)} images; corrupt={len(corrupt)}")
print("Counts:", {s: {c: len(files[s][c]) for c in CLASSES} for s in SPLITS})
print("Archives:", archive_results)
print("Overlap:", {f"{a}/{b}": {k: len(v) for k, v in ov.items()} for (a,b),ov in pair_overlap.items()})
