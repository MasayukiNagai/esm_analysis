# Protein analysis using ESM

## Usage

0. Get the repository and install dependencies

Download the repo:
```
git clone git@github.com:MasayukiNagai/esm_analysis.git
```

Install the dependencies.
The easiest way is to use `uv` (`module load UV` if it's available via the system).
```
# uv
uv sync

# or pip in an existing virtual environment
pip install .
```

If a kernel is needed for Jupyter notebook, run the following:
```
source .venv/bin/activate
python -m ipykernel install --user --name esm2-analysis --display-name "Python (esm2)"
```
