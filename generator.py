import os
import argparse
from data_loading import config
from main import pipeline

root_dir = os.path.dirname(os.path.abspath(__file__))
def generate_results(subjects, modalities, modes, override=False, verbose=False):
    results = {} 
    for modality in modalities:
        for mode in modes:
            print("="*60)
            print(f"Processing mode: {mode} | Modality: {modality}")
            print("-"*60)
            try:
                results[mode] = pipeline(
                    subjects,
                    modality,
                    mode,
                    config.STORIES,
                    config.NUIS_LISTENING,
                    config.NUIS_READING,
                    root_dir,
                    output_dir=root_dir / config.OUTPUT_DIR / "results_actual",
                    contexts_file = root_dir/ config.CONTEXTS_FILE,
                    embeddings_file = root_dir/ config.EMBEDDINGS_FILE,
                    overwrite_embeddings=override,
                    overwrite_contexts=override,
                    override = override,
                    verbose= verbose,
                )
            except Exception as e:
                print(f"Error processing mode {mode} and modality {modality}: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results from pipeline")
    parser.add_argument("--subjects", nargs="+", required=True, help="List of subjects")
    parser.add_argument("--modalities", nargs="+", required=True, help="List of modalities")
    parser.add_argument("--modes", nargs="+", required=True, help="List of modes")
    parser.add_argument("--override", action="store_true", help="Override existing files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    results = generate_results(
        subjects=args.subjects,
        modalities=args.modalities,
        modes=args.modes,
        override=args.override,
        verbose=args.verbose,
    )