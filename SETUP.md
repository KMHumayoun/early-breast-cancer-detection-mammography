# Install and run

Use 64-bit Python 3.12. The Mac benchmark used an Apple M1. The code selects CUDA when available, then Apple Metal (MPS), then CPU. A machine with an SSD for the official images and a supported GPU is preferable for the full run. The small fixture tests use CPU and do not download any patient images or pretrained weights.

## Get the data

Use the official noncommercial RSNA release and follow its access terms. The repository contains code and aggregate counts. Keep images, labels and local predictions outside version control. Extract the official training files to a folder containing `train.csv` and `train_images/<patient_id>/<image_id>.dcm`. The official Kaggle page lists 314.72 GB. Allow extra space if you keep the archive, extracted images and training checkpoints on the same drive. [RSNA access portal](https://mira.rsna.org/dataset/3) · [Official challenge data page](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)

## Windows

Open PowerShell in the project folder. These commands use the environment’s interpreter directly.

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python -m pytest -q
```

In the commands below, replace `python` with `& $python`. Set the path to your extracted release, for example `$data = "D:/rsna"`. Python and PowerShell accept forward slashes in the example paths.

## macOS

Install Python 3.12 and open Terminal in the project folder. Apple Silicon can use Metal when supported by the installed PyTorch build.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Ubuntu and Linux

On Ubuntu 24.04, install `python3.12` and `python3.12-venv` if needed. Create the environment as above. For a CUDA machine, install a compatible PyTorch build that matches the pinned PyTorch and torchvision releases before installing the rest. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

## Kaggle or another existing notebook workspace

Attach the official RSNA competition dataset to an authorized workspace. Use writable storage for this repository and its `runs/` directory. Set the dataset path to the mounted folder containing `train.csv` and `train_images`. The competition’s hidden test set is unnecessary. The default real training run needs the official torchvision ImageNet weights in its cache or internet access on first use. Check the workspace time quota and keep each completed model directory on persistent storage.

## Run the experiment

Replace `/path/to/rsna` with the actual folder. Run each command from the repository root. Preparation checks the labels and verifies that selected files exist. The resulting audit reports the actual number of patients, breasts and cancers per split.

```bash
python -m mammography prepare --csv "/path/to/rsna/train.csv" --images "/path/to/rsna/train_images" --output runs/index.json
python -m mammography train --model single --images "/path/to/rsna/train_images" --output runs/single
python -m mammography train --model paired --images "/path/to/rsna/train_images" --output runs/paired
python -m mammography evaluate --images "/path/to/rsna/train_images" --single runs/single --paired runs/paired --output runs/evaluation
```

Each training run writes its best checkpoint and a per-epoch history. The completion record is written after the final epoch. Choose fresh output directories for new attempts. Evaluation accepts only completed runs with matching manifests, resolution, initialization, training seed and epoch budget. It writes `metrics.json`, `heldout-site.png` and a private patient-level predictions CSV. Review only aggregate results for publication.

To measure compute before fetching images, run `python -m mammography benchmark`. This uses generated tensors and excludes DICOM loading and validation. Once training starts, `history.json` gives actual epoch wall time including reading images and validation. The full evaluation includes independent-site inference and a patient bootstrap.

## Recreate the published cohort chart

Run `python scripts/plot_cohort.py`. It reads the published two-site counts under `reports/`. The model diagram can be rebuilt with `python scripts/plot_model.py`.
