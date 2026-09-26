# Understanding Grammar Representation in the Human Brain via Information-Restricted LLMs

Bachelor's thesis project analyzing fMRI responses to language stimuli using encoding
models built on information-restricted LLM features. The full write-up is in
[`docs/thesis.tex`](docs/thesis.tex) (compiled PDF at
[`docs/LaTeX_out/thesis.pdf`](docs/LaTeX_out/thesis.pdf)).

## Repository layout

- [`main.py`](main.py) — CLI entry point that runs the encoding-model pipeline
  (feature preparation, ridge regression, scoring) for a given subject/modality.
- [`data_loading/`](data_loading) — package for loading and preprocessing stimuli,
  fMRI responses, embeddings, and TR files (`load_data`, `prepare_data`, etc., see
  [`data_loading/__init__.py`](data_loading/__init__.py) for the public API).
- [`scripts/`](scripts) — analysis and plotting scripts/notebooks
  (`quantitative.py`, `plotting_utils.py`, `plotting.ipynb`, `explorer.ipynb`).
- [`quantitative2.py`](quantitative2.py) — a newer, in-progress variant of
  [`scripts/quantitative.py`](scripts/quantitative.py); the two have diverged and
  have not yet been reconciled into a single canonical script.
- `data/` — raw stimuli, responses, textgrids, and mapper files (gitignored; not
  included in the repo).
- `outputs/` — pipeline results (`results_actual/`, `results_control/`, one `.npy`
  per subject/modality).
- `docs/` — thesis LaTeX source, figures, reference literature, and the compiled
  build output (`docs/LaTeX_out/`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some dependencies (`pycortex`, `himalaya`, `torch`) may need extra system setup
(CUDA, a pycortex filestore config) depending on your machine.

## Running the pipeline

```bash
python main.py --subject subject07 --modality listening --mode baseline
```

See `python main.py --help` for all options (subject, modality, feature mode,
TR/transcript directories, verbosity).

## Status

Research code for an active thesis project — expect scripts to be exploratory
rather than production-quality, and interfaces to change as the analysis evolves.
