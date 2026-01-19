import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_loading import load_data, config
from main import pipeline
import numpy as np


SUBJECT = "subject07"
MODALITY = "listening"


def _run_mode_for_story(mode: str, story: str):
    R_trn, R_val, F_trn, F_val, stories_trn, stories_val = load_data(
        [SUBJECT],
        MODALITY,
        mode,
        [story, config.STORIES[-1]],
        config.FEATURE_PATH,
        config.RESPONSE_PATH,
        config.DATASEQ_PATH,
        config.CONTEXTS_FILE,
        config.EMBEDDINGS_FILE,
        overwrite_embeddings=False,
        overwrite_contexts=False,
        verbose=True,
    )

    use_keys = [mode] + (
        config.NUIS_LISTENING if MODALITY == "listening" else config.NUIS_READING
    )

    r, r2, r_sig, score = pipeline(
        R_trn[SUBJECT],
        R_val[SUBJECT],
        F_trn,
        F_val,
        stories_trn,
        stories_val,
        use_keys,
    )
    return story, mode, float(score), np.nanmean(r_sig)  # Return r_sig mean


def _write_results_csv(results, stories):
    out_dir = config.OUTPUT_DIR / "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir / f"{SUBJECT}_{MODALITY}_english1000_vs_baseline.csv"

    with open(csv_path, "w") as f:
        f.write("story,english1000,baseline\n")
        for s in stories:
            eng = results.get(s, {}).get("english1000", float("nan"))
            base = results.get(s, {}).get("baseline", float("nan"))
            f.write(f"{s},{eng},{base}\n")

    print(f"Saved comparison scores to {csv_path}")


def _print_results_table(results, stories):
    header = f"{'':12} {'english1000':>14} {'baseline':>14}"
    print(header)
    for s in stories:
        eng = results.get(s, {}).get("english1000", float("nan"))
        base = results.get(s, {}).get("baseline", float("nan"))
        print(f"{s:12} {eng:14.6f} {base:14.6f}")


def run_single_story(story: str):
    """Run english1000 and baseline for one story and update the CSV."""

    modes = ["english1000", "baseline"]
    results = {}
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=mp_ctx) as ex:
        future_to_mode = {
            ex.submit(_run_mode_for_story, mode, story): mode for mode in modes
        }
        for fut in as_completed(future_to_mode):
            mode = future_to_mode[fut]
            _, _, score, r_sig_mean = fut.result()
            results.setdefault(story, {})[mode] = score
            print(f"Mode: {mode}, Story: {story}, r_sig mean: {r_sig_mean}")

    # Load existing CSV if present to keep previous rows
    out_dir = config.OUTPUT_DIR / "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir / f"{SUBJECT}_{MODALITY}_english1000_vs_baseline.csv"
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            lines = f.read().strip().splitlines()
        for line in lines[1:]:
            story_id, eng, base = line.split(",")
            existing.setdefault(story_id, {})["english1000"] = float(eng)
            existing[story_id]["baseline"] = float(base)

    existing.update(results)
    all_stories = sorted(existing.keys())
    _write_results_csv(existing, all_stories)
    _print_results_table(existing, all_stories)


def main():
    stories = config.STORIES[:-1]

    results = {}
    for story in stories:
        # Process each story sequentially
        run_single_story(story)  # This will handle modes in parallel

    _print_results_table(results, stories)
    _write_results_csv(results, stories)


if __name__ == "__main__":
    main()
