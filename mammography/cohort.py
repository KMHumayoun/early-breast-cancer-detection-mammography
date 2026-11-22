"""Build a private breast-level index from the official labeled RSNA release."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


def build_cohort(metadata, dicoms, output, seed=83, development_site="1"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output file: {output}")
    groups = defaultdict(list)
    patient_site = {}
    image_ids = set()
    with Path(metadata).open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"patient_id", "image_id", "site_id", "laterality", "view", "cancer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing header fields: {required.difference(reader.fieldnames or [])}")
        for row in reader:
            pid, image = row["patient_id"], row["image_id"]
            side, site = row["laterality"], row["site_id"]
            if not pid.isdecimal() or not image.isdecimal() or side not in {"L", "R"}:
                raise ValueError("Invalid RSNA image identifier or laterality")
            if site not in {"1", "2"} or row["cancer"] not in {"0", "1"}:
                raise ValueError("Expected two RSNA sites and binary training labels")
            if image in image_ids or patient_site.setdefault(pid, site) != site:
                raise ValueError("Duplicate image ID or patient at multiple sites")
            image_ids.add(image)
            groups[pid, side].append(row)
    breasts = []
    omissions = Counter()
    for (pid, side), rows in sorted(groups.items()):
        labels = {r["cancer"] for r in rows}
        if len(labels) != 1:
            raise ValueError("Conflicting labels within a breast")
        entry = dict(patient=pid, side=side, site=patient_site[pid], cancer=int(labels.pop()))
        for view in ("CC", "MLO"):
            choices = sorted(int(r["image_id"]) for r in rows if r["view"] == view)
            if choices:
                # Stable across input order, with no access to pathology or image content.
                entry[view] = f"{pid}/{choices[0]}.dcm"
                omissions["extra_standard_images"] += len(choices) - 1
        if "CC" not in entry or "MLO" not in entry:
            omissions[f"incomplete_pairs_label_{entry['cancer']}"] += 1
            continue
        for view in ("CC", "MLO"):
            if not (Path(dicoms) / entry[view]).is_file():
                raise FileNotFoundError(f"Missing selected {view} image under the DICOM root")
        breasts.append(entry)
    patient_label = defaultdict(int)
    for b in breasts:
        patient_label[b["patient"]] = max(patient_label[b["patient"]], b["cancer"])
    ids = sorted(p for p in patient_label if patient_site[p] == development_site)
    def partition(items, size):
        return train_test_split(items, test_size=size, random_state=seed,
                                stratify=[patient_label[p] for p in items])
    train, reserved = partition(ids, 0.30)
    selection, calibration = partition(reserved, 0.5)
    split = {p: group for group, patients in [("train", train), ("select", selection),
                                               ("calibrate", calibration)] for p in patients}
    for b in breasts:
        b["split"] = split.get(b["patient"], "test")
    counts = {}
    for group in ("train", "select", "calibrate", "test"):
        subset = [b for b in breasts if b["split"] == group]
        cases = sum(b["cancer"] for b in subset)
        if not 0 < cases < len(subset):
            raise ValueError(f"{group} needs both outcomes after view filtering")
        counts[group] = dict(patients=len({b["patient"] for b in subset}), breasts=len(subset),
                             cancers=cases)
    audit = dict(csv_sha256=hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
                 source_images=len(image_ids), source_breasts=len(groups), seed=seed,
                 development_site=development_site, exclusions=dict(omissions), splits=counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as fp:
        json.dump(dict(audit=audit, breasts=breasts), fp)
    return audit
