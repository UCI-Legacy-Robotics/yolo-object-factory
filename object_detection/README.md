# Object Detection — URC 2026

YOLOv8/v11 object detection pipeline for the University Rover Challenge.

## Scope

This lives in its own repository, separate from the year's rover/ROS codebase.
The pipeline is dataset- and year-agnostic — model, dataset, and every training
argument come from config — so it carries forward to future URC seasons instead
of being re-forked each year. Keeping it standalone also means training on HPC3,
or on any machine that just needs to produce a model, does not require cloning
the ROS stack.

Consume it from a year's rover repo as a git submodule (or just use the exported
`.onnx`/`.engine` artifacts); the deployment-side code belongs in that repo, not
here.

## Directory Structure

```
object_detection/
├── .gitignore
├── .env.example
├── README.md
├── requirements.txt
├── __init__.py
├── data/
│   ├── data.py                 # Roboflow download logic
│   ├── load_data.ipynb         # exploration / sanity-check notebook
│   ├── data.yaml               # stable train/val/test descriptor
│   └── URC-2024-.../           # (gitignored) actual dataset
│       └── data.yaml
├── src/
│   ├── train.py                # training entry point
│   ├── export.py               # .pt → ONNX → TensorRT engine
│   ├── evaluate.py             # held-out test-set metrics
│   ├── failure_analysis.py     # annotated FP/FN/low-confidence cases
│   ├── benchmark.py            # latency and FPS benchmarking
│   ├── dataset_validation.py   # dataset integrity/leakage checks
│   └── inference.py            # load .engine, run detection (Jetson/ROS2)
├── scripts/
│   ├── download_data.slurm     # HPC3 job: fetch dataset
│   ├── run_hpc3.slurm          # HPC3 job: train model
│   └── tune_hpc3.slurm         # HPC3 job: evolutionary tuning
├── configs/
│   └── train_config.yaml       # model factory input, dataset path, train args
├── logs/                       # SLURM stdout/stderr
├── runs/                       # (gitignored) training outputs / weights
└── models/                     # (gitignored) exported .onnx / .engine
```

## Quick Start

1. Copy `.env.example` → `.env` and add your Roboflow API key.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset: `python data/data.py`
4. Validate the dataset: `python src/dataset_validation.py`
5. Train: `python src/train.py --config configs/train_config.yaml`
6. Evaluate on the held-out test split: `python src/evaluate.py --weights runs/yolo11s_baseline/weights/best.pt`
7. Export for Jetson: `python src/export.py --weights runs/yolo11s_baseline/weights/best.pt --format engine --half`

## Model Factory

`src/train.py` builds YOLO models through `src/yolo_model_factory.py`.
The factory keeps model selection, dataset paths, training arguments, and
tuning arguments in config instead of hardcoding them in Python.

Minimum full config:

```yaml
model:
  type: yolo
  source: yolo11s.pt
  task: detect
data: data/data.yaml
```

The `model` entry can be either a shorthand string or a structured mapping:

```yaml
model: yolo11s.pt
```

```yaml
model:
  type: yolo
  source: yolo11s.pt
  task: detect
```

Supported source aliases are `source`, `weights`, `checkpoint`, `cfg`, and
`architecture`. You can also build the source name from family and size:

```yaml
model:
  type: yolo
  family: yolo11
  size: n
  suffix: pt
```

Local model files are resolved from the current directory first, then from
`object_detection/`. If the file is not found locally, the source string is
left untouched so Ultralytics can resolve or download it. Dataset paths are
resolved the same way, and passing a dataset directory automatically uses its
`data.yaml`.

## Training Configuration

Train from the default config:

```bash
cd object_detection
python src/train.py --config configs/train_config.yaml
```

From the repository root, the module entry point also works:

```bash
python -m object_detection.src.train --config configs/train_config.yaml
```

Switch model size/checkpoint or dataset from the terminal:

```bash
python src/train.py --model yolo11n.pt --data /path/to/dataset/data.yaml --epochs 100 --device 0
python src/train.py --data /path/to/dataset_dir --name yolo11n_custom_dataset
```

Any Ultralytics `train()` argument can be overridden with `--set`. Dot notation
can also update nested model or train config keys:

```bash
python src/train.py --set lr0=0.005 --set mosaic=0.5 --set optimizer=SGD
python src/train.py --set model.source=yolo11m.pt --set model.task=detect
python src/train.py --set train.project=runs --set train.name=yolo11m_trial
```

Common shortcut flags such as `--epochs`, `--batch`, `--imgsz`, `--device`,
`--workers`, `--project`, `--name`, and `--resume` are merged into the training
arguments before calling Ultralytics.

Run hyperparameter tuning from the same config:

```bash
python src/train.py --tune --iterations 30 --tune-epochs 30
```

Ultralytics tuning writes a hyperparameter-only `best_hyperparameters.yaml`.
You can use that file directly as `--config`; the training script overlays it
on top of `configs/train_config.yaml` so the model and dataset settings are
preserved:

```bash
python src/train.py --config runs/tune/best_hyperparameters.yaml --name yolo11s_tuned_final
```

If a config includes only one of the required `model` or `data` keys, it is
treated as malformed and fails before training. Include both keys for a full
config, or omit both for a tuned-parameters overlay.

## Evaluation Artifacts

Evaluate a trained checkpoint on the held-out test split:

```bash
python src/evaluate.py --weights runs/yolo11s_baseline/weights/best.pt
```

Metrics are written to `runs/evaluation/metrics.json` and
`runs/evaluation/per_class_metrics.csv`. Validation plots, including current
Ultralytics `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, and
`BoxR_curve.png` files, are copied into `runs/evaluation/plots/`.

`results.png` from the training run that produced the weights is copied in as
well. It charts train and val loss per epoch, which is where overfitting shows
up — training loss still falling while validation loss turns back upward. Since
`model.val()` does not regenerate that plot, it is read from the run directory
above `--weights` (`runs/<name>/results.png`) and skipped if that directory is
no longer around.

## HPC3 Usage

HPC3 is a SLURM cluster: you do not run training on the login node, you submit a
job script that requests a GPU and runs for you when resources free up.

**Cluster documentation:** <https://rcic.uci.edu/hpc3/> — the RCIC docs cover
account access, `module` usage, SLURM submission, partitions, and storage
quotas. Read the SLURM and GPU sections before your first submission.

### First-time setup on the cluster

```bash
ssh <ucnetid>@hpc3.rcic.uci.edu
git clone <this-repo-url> && cd <repo>/object_detection

module load python/3.10 cuda/12.1     # confirm versions with `module avail`
python -m venv ~/.venvs/urc           # the job scripts activate this path
source ~/.venvs/urc/bin/activate
pip install -r requirements.txt

cp .env.example .env                  # add your Roboflow API key
```

If you put the venv somewhere else, update the `source ~/.venvs/urc/...` line in
each script under `scripts/`.

### Submitting jobs

```bash
cd object_detection
sbatch scripts/download_data.slurm   # download dataset on HPC3
sbatch scripts/run_hpc3.slurm        # submit training job
sbatch scripts/tune_hpc3.slurm       # submit tuning job
```

Pass training overrides through SLURM after the script name:

```bash
sbatch scripts/run_hpc3.slurm --model yolo11m.pt --data /path/to/data.yaml --epochs 200
sbatch scripts/tune_hpc3.slurm --set lr0=0.005 --iterations 40
```

### Monitoring and results

```bash
squeue -u $USER                      # your queued/running jobs
scancel <jobid>                      # cancel a job
tail -f logs/train_<jobid>.out       # live training output
sinfo -s                             # partitions you can actually submit to
```

Job stdout/stderr land in `logs/`, and training artifacts in `runs/<name>/`.
Pull results back to your machine with
`scp -r <ucnetid>@hpc3.rcic.uci.edu:<path>/object_detection/runs/<name> .`

### Partitions

`run_hpc3.slurm` targets `gpu` and `tune_hpc3.slurm` targets `free-gpu`. The
free partition is cheaper but **preemptible** — a job there can be killed to make
room for an allocated job, which is why the configs set `save_period` so a
restart resumes from the last checkpoint instead of from scratch. Tuning is the
unattended workload that suits it; a long final training run generally does not.

Verify the partition names with `sinfo` before submitting and adjust the
`#SBATCH --partition=...` lines if they differ — a wrong name bounces the job.
