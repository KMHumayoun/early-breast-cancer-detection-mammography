"""All test inputs are generated locally; no RSNA patients are used."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammography.cohort import build_cohort
from mammography.images import read_image
from mammography.runtime import train
from mammography.evaluation import evaluate, operating_point


def dicom(path, photo="MONOCHROME2"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = 64, 48
    ds.PhotometricInterpretation = photo
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 1,16,12,11,0
    arr = np.arange(64*48, dtype="<u2").reshape(64,48)
    ds.PixelPaddingValue = 4095
    arr[:3] = ds.PixelPaddingValue
    ds.PixelData = arr.tobytes()
    ds.save_as(path,enforce_file_format=True)


def test_dicom_polarity_padding_and_aspect(tmp_path):
    for mode in ["MONOCHROME1","MONOCHROME2"]:
        dicom(tmp_path/mode, mode)
    a, b = [read_image(tmp_path/p,96) for p in ["MONOCHROME1","MONOCHROME2"]]
    assert a.shape == (1,96,64)
    assert a[:,0].max() == b[:,0].max() == 0
    assert a[0,24,24] > a[0,65,24] and b[0,24,24] < b[0,65,24]


def test_cutoff_with_ties():
    negatives = np.array([.2]*8 + [.7]*2)
    cutoff = operating_point(negatives)
    assert (negatives < cutoff).mean() >= .90


def fixture_inputs(folder):
    import os
    template = folder/"template.dcm"
    dicom(template)
    fields = ["patient_id","site_id","laterality","image_id","view","cancer","biopsy"]
    path = folder/"labels.csv"
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pid in range(48):
            location = folder/str(pid)
            location.mkdir()
            for j,(side,view) in enumerate([("L","CC"),("L","MLO"),("R","CC"),("R","MLO")]):
                image=10*pid+j
                os.link(template,location/f"{image}.dcm")
                writer.writerow(dict(patient_id=pid,site_id=1 if pid<40 else 2,laterality=side,
                                image_id=image,view=view,cancer=int(pid%2==0 and side=="L"),biopsy=1))
    return path


def test_patient_split_views_and_validation(tmp_path):
    path=fixture_inputs(tmp_path)
    index=tmp_path/"index.json"
    audit=build_cohort(path,tmp_path,index)
    breasts=json.loads(index.read_text())["breasts"]
    assert audit["splits"]["test"]["patients"]==8
    assert all(len({r["split"] for r in breasts if r["patient"]==p})==1 for p in {r["patient"] for r in breasts})
    assert all((r["site"]=="2") == (r["split"]=="test") for r in breasts)
    assert all("biopsy" not in r for r in breasts)
    with path.open("a") as f:
        f.write("0,1,L,0,CC,1,1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        build_cohort(path,tmp_path,tmp_path/"bad.json")


def test_training_to_paired_evaluation(tmp_path, monkeypatch):
    # Tiny fixtures deliberately have identical pixels. Scores have no medical meaning.
    import mammography.runtime as runtime
    import mammography.evaluation as evaluation
    monkeypatch.setattr(runtime,"device",lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluation,"device",lambda: torch.device("cpu"))
    torch.set_num_threads(2)
    meta=fixture_inputs(tmp_path)
    index=tmp_path/"index.json"
    build_cohort(meta,tmp_path,index)
    args=SimpleNamespace(index=index,images=tmp_path,height=64,batch=4,workers=0,seed=83,
                         random_init=True,epochs=1,bootstrap=12)
    for name in ("single","paired"):
        args.model=name
        args.output=tmp_path/name
        train(args)
        setattr(args,name,args.output)
    args.output=tmp_path/"evaluation"
    report=evaluate(args)
    assert report["test_breasts"] == 16 and report["test_patients"] == 8
    assert (args.output/"heldout-site.png").stat().st_size > 0
    assert all(0 <= v["average_precision"] <= 1 for v in report["metrics"].values())
