#!/usr/bin/env python3
"""
One-time script to create DataSequence objects from TR files and TextGrids.
Saves them as pickle files for later use.
"""
import argparse
import pickle
from pathlib import Path
import sys
from typing import List
import os

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading import DataSequence, TRFile, config

BAD_WORDS = [
    "{br}",
    "{lg}",
    "{ls}",
    "{ns}",
    "{cg}",
    "",
    "sp",
    "sentence_start",
    "sentence_end",
]

STORIES = config.STORIES


def create_data_sequences(
    stories: List[str],
    trfile_dir: str,
    textgrid_dir: str,
    dataseq_dir: str,
    bad_words: List[str] = BAD_WORDS,
    word_time: str = "middle",
) -> dict:
    """
    Create DataSequence objects from TR files and TextGrid textgrids.

    Args:
        stories: List of story identifiers (e.g., ['story1', 'story2'])
        trfile_dir: Directory containing TR files
        textgrid_dir: Directory containing TextGrid textgrid files
        dataseq_dir: Directory to save DataSequence pickle files
        bad_words: List of words to remove from sequences
        word_time: When to timestamp each word ('start', 'middle', or 'end')
    """
    dataseqs = {}

    # Load TR files and textgrids
    print("Loading TR files...", end=" ")
    trfiles = _load_trfiles(stories, trfile_dir)
    print("✓")

    print("Loading TextGrid textgrids...", end=" ")
    textgrids = _load_textgrids(stories, textgrid_dir)
    print("✓")

    # Create DataSequences for each story
    for story in tqdm(stories, total=len(stories), desc="Creating DataSequences"):
        # Create initial DataSequence from TextGrid and TR file
        ds = DataSequence.from_grid(
            textgrids[story], trfiles[story], word_time=word_time
        )

        # Clean bad words
        text = np.array(ds.data)
        bad_words_indices = np.where(np.isin(text, bad_words))[0]

        if len(bad_words_indices) > 0:
            # Remove bad words from data
            cleaned_data = np.delete(text, bad_words_indices).tolist()

            # Remove corresponding timestamps
            if ds.data_times is not None:
                cleaned_times = np.delete(ds.data_times, bad_words_indices)
            else:
                cleaned_times = None

            # Adjust split indices (decrement for each removed word before split)
            split_inds_array = np.array(ds.split_inds)
            for index in sorted(bad_words_indices, reverse=True):
                split_inds_array[split_inds_array > index] -= 1

            # Create cleaned DataSequence
            ds = DataSequence(
                cleaned_data, split_inds_array.tolist(), cleaned_times, ds.tr_times
            )

        with open(os.path.join(dataseq_dir, story + ".pkl"), "wb") as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Print summary for first story
        if story == stories[0]:
            print(f"\n  Example ('{story}'):")
            print(f"    Words: {len(ds.data)}")
            print(f"    TRs: {len(ds.split_inds) + 1}")
            print(f"    Bad words removed: {len(bad_words_indices)}")
            print(f"    First 10 words: {ds.data[:10]}")


def _load_trfiles(stories, tr_dir):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.

    Parameters:
    -----------
    stories : list
        A list of stories
    tr_dir : str
        Local path where trfiles are stored.

    Returns:
    --------
    trdict : dict
        A dictionary of TRFile objects for each stories entry.
    """

    trdict = dict()
    for story in stories:
        try:
            trf = TRFile(os.path.join(tr_dir, story + ".report"))
            trdict[story] = trf
        except Exception as e:
            print(e)
    return trdict


def _load_textgrids(stories, tg_dir):
    """Loads TextGrid files and extracts word timing information.

    Parameters:
    -----------
    stories : list
        A list of TextGrid stories (without extension)
    tg_dir : str
        Local path where TextGrid files are stored.

    Returns:
    --------
    textgrid_dict : dict
        A dictionary mapping each story to a list of (start, end, word) tuples.
    """
    textgrid_dict = {}

    for story in tqdm(stories, desc="Loading TextGrid textgrids"):
        filepath = os.path.join(tg_dir, story)
        if not filepath.endswith(".TextGrid"):
            filepath += ".TextGrid"

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        words = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the line containing "word"
        word_index = -1
        for i, line in enumerate(lines):
            if '"word"' in line or 'story = "word"' in line:
                word_index = i
                break

        if word_index == -1:
            print(f"Warning: 'word' tier not found in {filepath}")
            continue

        # Skip the next 3 lines (metadata: start, end, count)
        start_index = word_index + 4

        # Parse each 3-row sequence as (start, end, word)
        i = start_index
        while i + 2 < len(lines):
            try:
                start = float(lines[i].strip())
                end = float(lines[i + 1].strip())
                word = lines[i + 2].strip().strip('"')
                words.append((start, end, word))
                i += 3
            except (ValueError, IndexError):
                # Stop if we can't parse the expected format
                break

        textgrid_dict[story] = words

    return textgrid_dict


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DataSequence objects from TR files and TextGrids"
    )
    parser.add_argument(
        "--dataseq-dir",
        required=True,
        help="Directory to save DataSequence pickle files",
    )
    parser.add_argument(
        "--trfile-dir", required=True, help="Directory containing TR files"
    )
    parser.add_argument(
        "--textgrid-dir",
        required=True,
        help="Directory containing TextGrid textgrid files",
    )
    parser.add_argument(
        "--word-time",
        choices=["start", "middle", "end"],
        default="middle",
        help="When to timestamp each word (default: middle)",
    )
    parser.add_argument(
        "--bad-words",
        nargs="*",
        default=BAD_WORDS,
        help="Words to remove from sequences",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DATA SEQUENCE PREPARATION")
    print("=" * 70)
    print(f"Stories: {len(STORIES)}")
    print(f"TR file directory: {args.trfile_dir}")
    print(f"textgrid directory: {args.textgrid_dir}")
    print(f"Word timestamp: {args.word_time}")
    print("=" * 70)

    # Create DataSequences
    wordseq = create_data_sequences(
        stories=STORIES,
        trfile_dir=args.trfile_dir,
        textgrid_dir=args.textgrid_dir,
        dataseq_dir=args.dataseq_dir,
        bad_words=args.bad_words,
        word_time=args.word_time,
    )

    # Summary
    print(f"\n✓ DataSequences saved to {args.dataseq_dir}\n")


if __name__ == "__main__":
    main()
