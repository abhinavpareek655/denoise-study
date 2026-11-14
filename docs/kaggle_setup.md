# Kaggle setup (each contributor has their own notebook and quota)

## Datasets (private)
| Slug (placeholder) | Content | Made by |
|---|---|---|
| `bsd-data` | folders `BSD68/` and `BSD400/` (from `bsd-data.zip`) | A, shared with B |
| `denoise-shared-vN` | published `artifacts/` (noisy PNGs, frozen tuning JSON, thresholds, checkpoints, CSVs) | each person per milestone |

Always pin the dataset **version number** when you add one as input; record it in the changelog.
`DENOISE_SHARED` takes an `os.pathsep`-separated list of read-only `artifacts/` roots; `artifact_path(..., must_exist=True)`
searches the writable `artifacts/` first, then these.

## Cell 1: bootstrap (every notebook)
```python
import os, subprocess
# Option A: Internet ON
subprocess.run("git clone https://github.com/<ORG>/denoise-study.git /kaggle/working/denoise-study", shell=True, check=True)
# Option B: Internet OFF -> add the repo as a dataset `denoise-code`, then:
# subprocess.run("cp -r /kaggle/input/denoise-code /kaggle/working/denoise-study", shell=True, check=True)
os.chdir("/kaggle/working/denoise-study")
os.environ["DENOISE_DATA"] = "/kaggle/input/bsd-data"                         # contains BSD68/ BSD400/
os.environ["DENOISE_SHARED"] = "/kaggle/input/denoise-shared-v1/artifacts"    # optional, read-only
import json; from common.versions import env_info; print(json.dumps(env_info(), indent=1))
```
Kaggle images already ship torch, numpy, scipy, scikit-image, pandas, PyYAML: do not `pip install`.

## Cell 2+: call stages (thin notebook)
```python
!python -m pytest                                           # first run: also exercises the torch tests
!python -m scripts.train --arch unet --variant P --seed 0 --bench 200   # throughput, then edit configs/train/train.yaml
!python -m scripts.train --arch unet --variant P --seed 0   # re-run the SAME command to resume after a session ends
!python -m scripts.infer --method unet --model-id unet_P_s0
!python -m scripts.evaluate --strict
```

## Rules
- Debug on CPU with tiny settings; switch the GPU accelerator on only for real runs; stop the session after.
- One GPU notebook at a time per account. Plan for sessions of 9 h or less (`max_hours` in the train config pauses safely).
- Save outputs: "Save Version" (Save & Run All) or download `artifacts/` and upload it as a dataset version.
- Optional: commit small files (CSV/JSON) from Kaggle with a GitHub token stored in Kaggle Secrets; never print the token.
- Runtime benchmarking: all methods in one session, same CPU, report medians, record the hardware string.
