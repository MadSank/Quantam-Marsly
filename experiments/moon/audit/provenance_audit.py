from __future__ import annotations

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import re
import argparse
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

PROJECT = Path(__file__).resolve().parents[3]
MOON = PROJECT / "data" / "raw" / "moon"
DATASET = MOON / "lrocnet"
MANIFEST = DATASET / "provenance_manifest.csv"
REPORT = MOON / "lrocnet_provenance_report.txt"
SPLITS = ("train", "val", "test")
CLASSES = ("ejecta", "oldcrater", "none")
IMAGE_EXTS = {".jpg", ".jpeg"}
NAME_RE = re.compile(r"^(?P<prefix>.+)_pyr-(?P<index>\d+)$", re.IGNORECASE)

# Exclusive creation is a final guard against replacing prior audit work.
parser = argparse.ArgumentParser()
parser.add_argument("--overwrite-output", action="store_true", help="Replace only this audit's own generated outputs")
args = parser.parse_args()
if (MANIFEST.exists() or REPORT.exists()) and not args.overwrite_output:
    raise FileExistsError(f"Refusing to overwrite existing outputs: {MANIFEST}, {REPORT}")

records: list[dict[str, object]] = []
corrupt: list[tuple[str, str]] = []
for split in SPLITS:
    for label in CLASSES:
        folder = DATASET / split / label
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            rel = path.relative_to(DATASET).as_posix()
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    width, height = im.size
                    mode = im.mode
                    rgb = im.convert("RGB")
                    exif_count = len(im.getexif())
                digest = sha256(path.read_bytes()).hexdigest()
                match = NAME_RE.fullmatch(path.stem)
                candidate_prefix = match.group("prefix") if match else ""
                candidate_suffix = match.group("index") if match else ""
                records.append({
                    "relative_image_path": rel,
                    "existing_split": split,
                    "class_label": label,
                    "width_px": width,
                    "height_px": height,
                    "color_mode": mode,
                    "sha256": digest,
                    # Semantics are deliberately not inferred from filenames alone.
                    "source_observation_id": "",
                    "source_image_id": "",
                    "site_id": "",
                    "filename_prefix_candidate_unverified": candidate_prefix,
                    "pyr_index_candidate_unverified": candidate_suffix,
                    "exif_field_count": exif_count,
                    "provenance_status": "FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED",
                    "provenance_notes": (
                        "Split/class from directory path; dimensions/mode/hash from file. "
                        "Local documentation does not establish filename prefix or _pyr index semantics; "
                        "source observation, source image, and geographic site remain unknown."
                    ),
                })
            except Exception as exc:
                corrupt.append((rel, f"{type(exc).__name__}: {exc}"))

if corrupt:
    # Fail closed: an incomplete manifest must not look complete.
    raise RuntimeError(f"Corrupt/unreadable files prevent complete manifest: {corrupt[:10]}")

expected_cols = [
    "relative_image_path", "existing_split", "class_label", "width_px", "height_px", "color_mode", "sha256",
    "source_observation_id", "source_image_id", "site_id", "filename_prefix_candidate_unverified",
    "pyr_index_candidate_unverified", "exif_field_count", "provenance_status", "provenance_notes",
]
if len(records) != 5000 or len({str(r["relative_image_path"]) for r in records}) != len(records):
    raise AssertionError(f"Expected 5,000 unique image records, got {len(records)}")

# Overlap analysis at each evidentiary level.
by_split: dict[str, list[dict[str, object]]] = {s: [r for r in records if r["existing_split"] == s] for s in SPLITS}
base_names: dict[str, set[str]] = {}
hashes: dict[str, set[str]] = {}
candidates: dict[str, set[str]] = {}
for split, recs in by_split.items():
    base_names[split] = {Path(str(r["relative_image_path"])).name.casefold() for r in recs}
    hashes[split] = {str(r["sha256"]) for r in recs}
    candidates[split] = {str(r["filename_prefix_candidate_unverified"]).casefold() for r in recs if r["filename_prefix_candidate_unverified"]}

cross_split = {}
for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        cross_split[(a, b)] = {
            "basename": base_names[a] & base_names[b],
            "sha256": hashes[a] & hashes[b],
            "candidate_prefix": candidates[a] & candidates[b],
        }

name_counts = Counter(Path(str(r["relative_image_path"])).name.casefold() for r in records)
hash_counts = Counter(str(r["sha256"]) for r in records)
duplicate_name_groups = {k: n for k, n in name_counts.items() if n > 1}
duplicate_hash_groups = {k: n for k, n in hash_counts.items() if n > 1}

candidate_groups: dict[str, dict[str, list[dict[str, object]]]] = {s: defaultdict(list) for s in SPLITS}
for r in records:
    prefix = str(r["filename_prefix_candidate_unverified"])
    if prefix:
        candidate_groups[str(r["existing_split"])][prefix.casefold()].append(r)

# Inspect ZIP listings and comments without extracting or writing to them.
archive_names = ["ejecta_train.zip", "oldcrater_train.zip", "none_train1.zip", "none_train2.zip",
                 "none_train3.zip", "none_train4.zip", "val.zip", "test.zip"]
archive_info = []
archive_members: dict[tuple[str, str], Counter] = defaultdict(Counter)
archive_targets = {
    "ejecta_train.zip": ("train", "ejecta"),
    "oldcrater_train.zip": ("train", "oldcrater"),
    "none_train1.zip": ("train", "none"), "none_train2.zip": ("train", "none"),
    "none_train3.zip": ("train", "none"), "none_train4.zip": ("train", "none"),
    "val.zip": ("val", None), "test.zip": ("test", None),
}
for archive in archive_names:
    split, fixed_class = archive_targets[archive]
    path = MOON / archive
    with ZipFile(path) as zf:
        nonimage_members = []
        image_count = 0
        for name in zf.namelist():
            normalized = name.replace("\\", "/")
            if normalized.lower().endswith(tuple(IMAGE_EXTS)) and "__MACOSX/" not in normalized and not Path(normalized).name.startswith("._"):
                image_count += 1
                parts = [part.casefold() for part in Path(normalized).parts]
                label = fixed_class or next((c for c in CLASSES if c in parts), None)
                if label:
                    archive_members[(split, label)][Path(normalized).name.casefold()] += 1
            elif not normalized.endswith("/") and "__MACOSX/" not in normalized and not Path(normalized).name.startswith("._"):
                nonimage_members.append(normalized)
        archive_info.append((archive, image_count, repr(zf.comment), nonimage_members))
archive_comparison = {}
for split in SPLITS:
    for label in CLASSES:
        archive_count = archive_members[(split, label)]
        extracted_count = Counter(Path(str(r["relative_image_path"])).name.casefold() for r in records
                                  if r["existing_split"] == split and r["class_label"] == label)
        archive_comparison[(split, label)] = {
            "archive_images": sum(archive_count.values()),
            "extracted_images": sum(extracted_count.values()),
            "missing": list((archive_count - extracted_count).elements()),
            "extra": list((extracted_count - archive_count).elements()),
        }

# Local metadata search results are recorded as scoped evidence, not external claims.
# The existing dataset audit has independently verified RGB channels are identical.
counts = {s: {c: sum(r["existing_split"] == s and r["class_label"] == c for r in records) for c in CLASSES} for s in SPLITS}
manifest_tmp = MANIFEST.with_suffix(".csv.tmp")
with manifest_tmp.open("x", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=expected_cols)
    writer.writeheader()
    writer.writerows(records)
manifest_tmp.replace(MANIFEST)

lines = [
    "LROCNet Lunar Dataset — Provenance and Split Audit",
    f"Dataset: {DATASET}",
    "Scope: local files only; no web access, downloads, training, image edits, or archive edits.",
    "",
    "Sources inspected:",
    "  - Existing data/raw/moon/lrocnet_dataset_audit.txt and data/raw/moon/lrocnet/audit_lrocnet.py.",
    "  - Repository README.md, pyproject.toml, source/tests via local text search for LROCNet, class names, filename pattern, ImageFolder, and classification utilities.",
    "  - The eight original ZIP archives: member paths, JPEG counts, comments, and non-image entries; membership was reconciled to split/class image basenames; no extraction performed.",
    "  - Extracted train/val/test folder paths, all image headers/decode, SHA-256 bytes, and JPEG EXIF fields.",
    "  - Existing _extract_staging file listing; only archive extraction artifacts such as .DS_Store/AppleDouble files were evident, not provenance or split documentation.",
    "",
    "Confirmed local facts:",
    f"  - Manifest contains {len(records)} unique relative image paths; all images decoded; dimensions/modes are derived from image files.",
    "  - The split and class are confirmed only as current directory placement; class semantic definitions are not documented locally.",
    f"  - {sum(bool(NAME_RE.fullmatch(Path(str(r['relative_image_path'])).stem)) for r in records)}/{len(records)} filenames match PREFIX_pyr-NNNN syntax (case-insensitive); this describes syntax, not the meaning of PREFIX or pyr-NNNN.",
    "  - All 5,000 JPEGs have no EXIF fields; the existing audit found identical RGB channels for all images (grayscale encoded as RGB).",
    "  - The ZIP archives have no comments. Their file members establish class/split membership, not how those splits were generated.",
    "",
    "Unresolved provenance:",
    "  - No local LROCNet README, citation, original source URL, class-definition document, source-image map, observation/site table, or split-generation manifest was found.",
    "  - Class names ejecta, oldcrater, and none are known as labels from folder/ZIP structure only. Their annotation criteria, ambiguity rules, and labeling process are unknown.",
    "  - Filename components are not semantically verified. The prefix before _pyr- is retained in the CSV only as filename_prefix_candidate_unverified; the numeric suffix is pyr_index_candidate_unverified. Neither is promoted to an observation, image, or site ID.",
    "  - Multiple files do share a candidate prefix, but without documentation it is not confirmed that these are patches from one source observation/image. No geographic/site IDs can be populated defensibly.",
    "  - The original split-generation methodology is unknown. Folder and archive names encode the resulting partitions, not random/grouped/site-wise sampling procedure.",
    "",
    "Candidate filename-prefix grouping (UNVERIFIED hypothesis only):",
]
for split in SPLITS:
    groups = candidate_groups[split]
    sizes = [len(v) for v in groups.values()]
    multi_class = sum(len({str(r["class_label"]) for r in v}) > 1 for v in groups.values())
    lines.append(f"  {split}: candidate groups={len(groups)}; images represented={sum(sizes)}; group sizes min/median/max={min(sizes) if sizes else 0}/{sorted(sizes)[len(sizes)//2] if sizes else 0}/{max(sizes) if sizes else 0}; groups containing multiple class labels={multi_class}.")
lines.extend([
    "  The candidate-prefix group sets have zero pairwise overlap across the supplied splits. This is evidence only about a literal filename-token rule; it does not prove source-observation or geographic separation.",
    "",
    "Overlap by evidentiary level:",
    f"  Duplicate basenames across all {len(records)} files: {len(duplicate_name_groups)} duplicate groups.",
    f"  Exact byte-identical SHA-256 groups: {len(duplicate_hash_groups)} duplicate groups.",
    "  Same source observation: UNKNOWN (no defensible source-observation IDs).",
    "  Same source image: UNKNOWN (no source-image map; candidate prefixes are unverified).",
    "  Same geographic site: UNKNOWN (no geolocation/site metadata).",
])
for (a, b), ov in cross_split.items():
    lines.append(f"  {a} vs {b}: basename overlap={len(ov['basename'])}; exact SHA-256 overlap={len(ov['sha256'])}; unverified candidate-prefix overlap={len(ov['candidate_prefix'])}.")
lines.extend(["", "Existing split / archive reconciliation:"])
for split in SPLITS:
    lines.append(f"  {split}: {sum(counts[split].values())} images; class counts={counts[split]}.")
for archive, n, comment, nonimage in archive_info:
    lines.append(f"  {archive}: {n} JPEG members; comment={comment}; ancillary non-image entries={nonimage}.")
for (split, label), comparison in archive_comparison.items():
    lines.append(f"  {split}/{label}: ZIP JPEGs={comparison['archive_images']}; extracted={comparison['extracted_images']}; missing={len(comparison['missing'])}; extra={len(comparison['extra'])}.")
    if comparison["missing"] or comparison["extra"]:
        lines.append(f"    mismatch examples: missing={comparison['missing'][:5]}; extra={comparison['extra'][:5]}.")
lines.extend(["  ZIP reconciliation verifies archive membership, not the split-generation algorithm.", "",
    "Split recommendation:",
    "  Do not replace the supplied train/validation/test split in this phase; do not create a new supposedly source-grouped/site-held-out split. The candidate prefix has group structure (including many multi-label groups), but its source semantics are unverified and geographic grouping metadata is absent.",
    "  For the coursework baseline, retain the supplied partitions, state that their creation method is unknown, and report results only as performance on this supplied split—not as site-independent generalization. Keep validation for model selection and the supplied test partition locked until final evaluation.",
    "  Before claiming source-image-grouped evaluation, obtain the dataset's authoritative documentation or source manifest and verify the filename mapping. If verified, group on the official source-image/observation ID and split groups deterministically with a fixed seed and class-balance constraints. For geographic/site generalization, obtain actual site/geolocation IDs; image-level grouping cannot substitute for site grouping.",
    "",
    "Manifest semantics:",
    "  source_observation_id, source_image_id, and site_id are blank by design. filename_prefix_candidate_unverified and pyr_index_candidate_unverified preserve the observed filename tokens without asserting what they mean. Provenance status and notes make the uncertainty explicit per row.",
    "",
    f"Manifest: {MANIFEST}",
    f"Report: {REPORT}",
    "No new split manifest was created because the required group semantics are not established.",
])
report_tmp = REPORT.with_suffix(".txt.tmp")
with report_tmp.open("x", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
report_tmp.replace(REPORT)

# Re-open generated files and assert one-to-one manifest coverage and blank unsupported IDs.
with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    manifest_rows = list(csv.DictReader(f))
assert len(manifest_rows) == 5000
assert len({r["relative_image_path"] for r in manifest_rows}) == 5000
assert all(not r["source_observation_id"] and not r["source_image_id"] and not r["site_id"] for r in manifest_rows)
assert sum(Counter(r["existing_split"] for r in manifest_rows).values()) == 5000
assert all(not x["missing"] and not x["extra"] and x["archive_images"] == x["extracted_images"]
           for x in archive_comparison.values())
assert REPORT.stat().st_size > 0
print(f"Manifest verified: {MANIFEST} ({len(manifest_rows)} rows)")
print(f"Report verified: {REPORT} ({REPORT.stat().st_size} bytes)")
print("Counts:", counts)
print("Candidate-prefix overlap:", {f"{a}/{b}": len(v["candidate_prefix"]) for (a,b),v in cross_split.items()})
