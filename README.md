# Understanding Grammar Representation in the Human Brain via Information-Restricted LLMs

Bachelor's thesis project analyzing fMRI responses to language stimuli using encoding
models built on information-restricted LLM features. The full write-up is in
[`docs/thesis.tex`](docs/thesis.tex) (compiled PDF at
[`docs/LaTeX_out/thesis.pdf`](docs/LaTeX_out/thesis.pdf)).

## Repository layout

- [`src/grammar_brain/main.py`](src/grammar_brain/main.py) — CLI entry point that
  runs the encoding-model pipeline (feature preparation, ridge regression,
  scoring) for a given subject/modality.
- [`src/grammar_brain/quantitative.py`](src/grammar_brain/quantitative.py) — ROI
  dominance analysis, produces the figures used in the thesis
  (`docs/figures/Fig4/`).
- [`data_loading/`](data_loading) — package for loading and preprocessing stimuli,
  fMRI responses, embeddings, and TR files (`load_data`, `prepare_data`, etc., see
  [`data_loading/__init__.py`](data_loading/__init__.py) for the public API).
- [`scripts/`](scripts) — analysis and plotting scripts/notebooks
  (`plotting_utils.py`, `plotting.ipynb`, `explorer.ipynb`).
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

`data_loading/data_sequence.py` also depends on `text_lite`, an internal Gallant
Lab package that isn't published on PyPI — it isn't in `requirements.txt` and
needs to be installed separately (e.g. `pip install -e /path/to/text_lite`) from
wherever you obtained it.

## Running the pipeline

```bash
python -m src.grammar_brain.main --subject subject07 --modality listening --mode baseline
```

Run as a module (`-m`) from the repo root so `data_loading` and `scripts` resolve
correctly. See `python -m src.grammar_brain.main --help` for all options (subject,
modality, feature mode, TR/transcript directories, verbosity).

## Tests

```bash
python -m pytest tests/
```

Covers `data_loading/` (the most reusable, decoupled part of the codebase).
The encoding pipeline in `src/grammar_brain/main.py` itself isn't covered — it
needs real fMRI data and, ideally, a GPU to exercise meaningfully.

## Status

Research code for an active thesis project — expect scripts to be exploratory
rather than production-quality, and interfaces to change as the analysis evolves.

## Acknowledgments

`data_loading/tr_file.py` and `data_loading/data_sequence.py` are adapted from
[HuthLab/deep-fMRI-dataset](https://github.com/HuthLab/deep-fMRI-dataset) (MIT
License, Copyright (c) 2023 HuthLab).
