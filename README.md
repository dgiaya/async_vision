A repository for working on metric reconstruction from unsychronized multi-camera rigs and other related problems.

## Environment
If you cloned without submodules, make sure to fetch them before creating the environment:
```sh
git submodule update --init --recursive
```

Create and activate the conda environment from the repo root:
```sh
conda env create -f environment.yaml
conda activate vision
```
This environment installs the local `calibration/libs/aprilgrid` package via pip.


