import numpy as np
import pickle
import os
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any


def load_dataseqs(dataseq_dir: str, stories: List[str]) -> Dict:
    """Load DataSequence objects from a pickle file."""
    dataseqs = {}
    for story in stories:
        with open(os.path.join(dataseq_dir, f"{story}.pkl"), "rb") as f:
            dataseqs[story] = pickle.load(f)
    return dataseqs


def load_contexts(contexts_file: str) -> Dict:
    """Load context data for different modes."""
    return np.load(contexts_file, allow_pickle=True).item()


def load_features(split: str, path: str, stories: List[str] = None) -> Dict:
    """Load training features."""
    data = dict()
    fname = os.path.join(path, f"features_{split}.hdf")
    with h5py.File(fname) as hf:
        for i, k in enumerate(tqdm(
            hf.keys(), desc=f"Loading features for split: {split}", leave=False
        )):
            if stories is not None:
                story_key = stories[i]
            else: 
                story_key = k
                
            data[story_key] = {}
            for j in tqdm(hf[k].keys(), desc=f"{k}", leave=False):
                data[story_key][j] = hf[k][j][()]
    return data


def load_responses(split: str, path: str, subjects: List[str], modality: str, stories: List[str] = None) -> Dict:
    """Load fMRI response data for specified subjects."""
    data = dict()
    for subject in subjects:
        try:
            fname = os.path.join(path, f"{subject}_{modality}_fmri_data_{split}.hdf")
            with h5py.File(fname) as hf:
                data[subject] = dict()
                for i, k in enumerate(tqdm(
                    hf.keys(), desc=f"Loading responses for {subject}", leave=False
                )):
                    if stories is not None:
                        story_key = stories[i]
                    else:
                        story_key = k
                    data[subject][story_key] = hf[k][()]
                    if data[subject][story_key].ndim == 3 and data[subject][story_key].shape[0] == 1:
                        data[subject][story_key] = data[subject][story_key][0]
        except OSError as e:
            print(f"Error loading data for subject {subject}: {e}")

    return data


def save_results(output_file: str, data: Any):
    """Save processed results to file."""
    np.savez(output_file, data=data)


def ensure_directory_exists(filepath: str):
    """Create parent directories if they don't exist."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
