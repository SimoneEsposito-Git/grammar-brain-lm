import os
from typing import Dict, List

import numpy as np
import h5py
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm

from data_sequence import DataSequence
from textgrid_utils import load_generic_trfiles, load_textgrid_transcripts
from sklearn.preprocessing import normalize

import context_utils as cu


# Utility function that may need to be defined or imported from elsewhere
def mapdict(d, func):
    """Apply a function to all values in a dictionary."""
    return {k: func(v) for k, v in d.items()}


# ============================================================================
# Response Processing
# ============================================================================


def load_responses(subjects, modality, split="trn", fdir="./"):
    """Load fMRI response data for specified subjects.

    Args:
        subjects: List of subject identifiers.
        modality: Type of fMRI modality to load.
        split: Data split ('trn', 'val', or 'tst'). Defaults to 'trn'.
        fdir: Directory containing response files. Defaults to './responses'.

    Returns:
        Dictionary mapping subjects to their response data.
    """
    data = dict()
    for subject in subjects:
        fname = os.path.join(
            fdir, "responses", f"{subject}_{modality}_fmri_data_{split}.hdf"
        )
        with h5py.File(fname) as hf:
            data[subject] = dict()
            for k in tqdm(
                hf.keys(), desc=f"Loading responses for {subject}", leave=False
            ):
                data[subject][k] = hf[k][()]

    return data


def stack_responses(R, stories, trim, standardize=True):
    """Stack response data across stories with optional standardization.

    Args:
        R: Dictionary of response data by story.
        stories: List of story identifiers to stack.
        trim: Number of initial time points to trim.
        standardize: Whether to z-score normalize. Defaults to True.

    Returns:
        Tuple of (stacked responses array, array of story lengths).
    """
    Ys, lens = [], []
    for s in stories:
        Y = np.asarray(R[s][trim:])
        if standardize:
            Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
        Ys.append(np.nan_to_num(Y))
        lens.append(Y.shape[0])
    return np.vstack(Ys), np.array(lens)


# ============================================================================
# Feature Processing
# ============================================================================


def load_features(split="trn", fdir="./") -> Dict:
    """Load feature data from HDF5 file.

    Args:
        split: Data split ('trn', 'val', or 'tst'). Defaults to 'trn'.
        fdir: Directory containing feature files. Defaults to './'.

    Returns:
        Dictionary containing loaded feature data.
    """
    data = dict()
    fname = os.path.join(fdir, "features", f"features_{split}_NEW.hdf")
    with h5py.File(fname) as hf:
        for k in tqdm(
            hf.keys(), desc=f"Loading features for split: {split}", leave=False
        ):
            data[k] = {}
            for j in tqdm(hf[k].keys(), desc=f"{k}", leave=False):
                data[k][j] = hf[k][j][()]
    return data


def prepare_features(
    F_trn,
    F_val,
    mode,
    stories,
    story_names,
    data_dir,
    trfile_dir,
    transcript_dir,
    overwrite_contexts=False,
    overwrite_embeddings=False,
    verbose=False,
    **kwargs,
):
    """
    Prepare features by loading or generating contextual embeddings.
    Args:
        F_trn: Training feature dictionary.
        F_val: Validation feature dictionary.
        mode: Context mode ('preceding', 'sentence', etc.).
        stories: List of story identifiers.
        story_names: List of story names for file lookup.
        data_dir: Directory to save/load contexts and embeddings.
        trfile_dir: Directory containing TR files.
        transcript_dir: Directory containing transcript files.
    Returns:
        Tuple of updated training and validation feature dictionaries.
    """

    if mode == "english1000":
        return F_trn, F_val

    wordseq = load_stimulus_word_sequences(
        stories, story_names, trfile_dir, transcript_dir
    )

    context_file = os.path.join(data_dir, "contexts.npz")
    embeddings_file = os.path.join(data_dir, "embeddings.npz")

    # Load or create contexts
    try:
        loaded_contexts = np.load(context_file, allow_pickle=True)
        # Convert NpzFile to plain dict
        contexts = {
            k: (
                loaded_contexts[k].item()
                if loaded_contexts[k].dtype == object
                else loaded_contexts[k]
            )
            for k in loaded_contexts.files
        }
    except FileNotFoundError:
        print(f"Warning: {context_file} not found. Creating new contexts.")
        contexts = {}
    except EOFError:
        print(f"Warning: {context_file} is empty or corrupted. Creating new contexts.")
        contexts = {}

    # Ensure mode exists
    if mode not in contexts or contexts[mode] is None or overwrite_contexts:
        contexts[mode] = {}

    # Generate contexts only for missing stories
    missing_ctx = [s for s in stories if s not in contexts[mode]]
    if len(missing_ctx) > 0:
        for story in missing_ctx:
            contexts[mode][story] = cu.generate_context(
                wordseq[story], mode, story=story, **kwargs
            )
        # Save updated contexts
        np.savez(
            context_file, **{k: np.array(v, dtype=object) for k, v in contexts.items()}
        )

    # Load or create embeddings
    try:
        loaded_embeddings = np.load(embeddings_file, allow_pickle=True)
        # Convert NpzFile to plain dict
        embeddings = {
            k: (
                loaded_embeddings[k].item()
                if loaded_embeddings[k].dtype == object
                else loaded_embeddings[k]
            )
            for k in loaded_embeddings.files
        }
    except FileNotFoundError:
        print(f"Warning: {embeddings_file} not found. Creating new embeddings.")
        embeddings = {}
    except EOFError:
        print(
            f"Warning: {embeddings_file} is empty or corrupted. Creating new embeddings."
        )
        embeddings = {}

    # Ensure mode exists
    if mode not in embeddings or embeddings[mode] is None or overwrite_embeddings:
        embeddings[mode] = {}

    # Generate embeddings only for missing stories
    missing_emb = [s for s in stories if s not in embeddings[mode]]
    if len(missing_emb) > 0:
        new_emb = contextual_embeddings(
            wordseq,
            "openai-community/gpt2-large",
            8,
            interp="lanczos",
            contexts=contexts[mode],
            verbose=verbose,
        )
        for s in missing_emb:
            embeddings[mode][s] = new_emb[s]
        # Save updated embeddings
        np.savez(
            embeddings_file,
            **{k: np.array(v, dtype=object) for k, v in embeddings.items()},
        )

    F_trn_ = F_trn.copy()
    F_val_ = F_val.copy()

    for story in stories[:-1]:
        # Only try to merge if the story exists in the target dictionary
        if story in F_trn_.keys():
            if mode not in F_trn_[story]:
                F_trn_[story][mode] = embeddings[mode][story]
            else:
                F_trn_[story][mode] = embeddings[mode][story]

    F_val_[stories[-1]][mode] = embeddings[mode][stories[-1]]
    return F_trn_, F_val_


def stack_features(F, stories, keys, standardize=True):
    """Stack selected features horizontally with optional standardization.

    Args:
        F: Feature dictionary by story.
        stories: List of story identifiers.
        keys: List of feature keys to stack.
        standardize: Whether to z-score normalize. Defaults to True.

    Returns:
        Dictionary mapping stories to stacked feature arrays.
    """
    blocks = {}
    for s in stories:
        X = np.hstack([np.asarray(F[s][k]) for k in keys])
        if standardize:
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        blocks[s] = X
    return blocks


def delay_features(X, stories, delays, circpad=False):
    """Create temporally delayed versions of features.

    Args:
        X: Dictionary of feature arrays by story.
        stories: List of story identifiers.
        delays: List or array of delay values (in time points).
        circpad: Whether to use circular padding. Defaults to False.

    Returns:
        Dictionary mapping stories to delayed feature arrays.
    """
    X_d = {}
    for s in stories:
        stim = X[s]
        nt, ndim = stim.shape
        dstims = []
        for di, d in enumerate(delays):
            dstim = np.zeros((nt, ndim))
            if d < 0:  ## negative delay
                dstim[:d, :] = stim[-d:, :]
                if circpad:
                    dstim[d:, :] = stim[:-d, :]
            elif d > 0:
                dstim[d:, :] = stim[:-d, :]
                if circpad:
                    dstim[:d, :] = stim[-d:, :]
            else:  ## d==0
                dstim = stim.copy()
            dstims.append(dstim)
        X_d[s] = np.hstack(dstims)
    return X_d


def stack_stories(X, stories):
    """Stack feature arrays across multiple stories.

    Args:
        X: Feature array or dictionary.
        stories: List of story identifiers.
        delays: Delay parameters (not used in current implementation).

    Returns:
        Vertically stacked array of all stories.
    """
    blocks = []
    for s in stories:
        blocks.append(X[s])
    return np.vstack(blocks)


def build_feature_groups(F_one_story, keys, n_delays):
    """Build group indices for delayed feature columns.

    Args:
        F_one_story: Feature dictionary for a single story.
        keys: List of feature keys.
        n_delays: Number of delay taps.

    Returns:
        Array of group indices for each column in delayed design matrix.
    """
    group_idx_raw = []
    for gi, k in enumerate(keys):
        group_idx_raw.append(np.full(F_one_story[k].shape[1], gi, dtype=int))
    group_idx_raw = np.hstack(group_idx_raw)
    groups_delayed = np.repeat(group_idx_raw, n_delays)
    return groups_delayed


def build_kernels_from_groups(X, groups_delayed):
    """Build separate kernel matrices for each feature group.

    Args:
        X: Feature design matrix.
        groups_delayed: Array of group indices for each column.

    Returns:
        3D array of kernel matrices stacked along first axis.
    """
    kernels = []
    unique_groups = np.unique(groups_delayed)

    for group in unique_groups:
        mask = groups_delayed == group
        X_group = X[:, mask]
        kernel = X_group @ X_group.T
        kernels.append(kernel)

    return np.stack(kernels, axis=0)


# =============================================================================
# Stimulus Processing
# =============================================================================


def load_stimulus_word_sequences(
    stories,
    names,
    trfile_dir,
    transcript_dir,
    bad_words: List[str] = [
        "{br}",
        "{lg}",
        "{ls}",
        "{ns}",
        "{cg}",
        "",
        "sp",
        "sentence_start",
        "sentence_end",
    ],
):
    """Load word sequences for all stories into DataSequence objects.

    Args:
        stories: List of story identifiers.
        names: List of story names for file lookup.
        trfile_dir: Directory containing TR files.
        transcript_dir: Directory containing transcript files.
        bad_words: List of words to remove from sequences.

    Returns:
        Dictionary mapping story identifiers to DataSequence objects.
    """
    wordseq = {}

    trfiles = load_generic_trfiles(names, trfile_dir)
    print("Loading transcripts...", end=" ")
    transcripts = load_textgrid_transcripts(names, transcript_dir)

    for story, name in zip(stories, names):
        ds = DataSequence.from_grid(transcripts[name], trfiles[name])

        # Remove bad words
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

            # Adjust split indices
            split_inds_array = np.array(ds.split_inds)
            for index in bad_words_indices[::-1]:
                split_inds_array[split_inds_array > index] -= 1

            # Create new cleaned DataSequence
            ds = DataSequence(
                cleaned_data, split_inds_array.tolist(), cleaned_times, ds.tr_times
            )

        wordseq[story] = ds

    return wordseq


def downsample(dsdict: Dict, interp: str = "mean"):
    """Downsamples each DataSequence in [dsdict] using the settings specified in the
    initializer.
    """
    # If each value in dict is another dict (e.g. when we get multiple layers of a contextual LM), downsample each dict separately.
    if type(list(dsdict.values())[0]) == dict:
        downsampled_dict = dict()
        for key in dsdict.keys():
            downsampled_dict[key] = mapdict(dsdict[key], lambda h: h.chunksums(interp))
        return downsampled_dict
    else:
        return mapdict(dsdict, lambda h: h.chunksums(interp))


def contextual_embeddings(
    wordseqs: dict,
    model_name: str,
    layer_num: int,
    contexts: dict,
    avg_tokens: bool = True,
    downsamp: bool = True,
    verbose: bool = False,
    interp: str = "lanczos",
):
    """Returns embeddings extracted from contextual models.

    args:
        layer_num: The layer from which to extract embeddings.
        model_name: Name of model to use (e.g. 'bert-base-multilingual-cased', 'xlm-mlm-xnli15-1024')
        avg_tokens: Take the average of the tokens over the window, instead of just the last one.
        downsample: If True, downsamples responses before returning.
    """
    # Get stimulus for stories.
    stimulus = dict()
    for stimulus_name, ds in list(wordseqs.items()):
        print(f"extracting {model_name} features for {stimulus_name}")
        stimulus[stimulus_name] = get_contextual_embeddings(
            ds=ds,
            model_name=model_name,
            layer_num=layer_num,
            contexts=contexts[stimulus_name],
            verbose=verbose,
        )
    if downsamp:
        return downsample(stimulus, interp=interp)
    else:
        return stimulus


def get_contextual_embeddings(
    ds: DataSequence,
    model_name: str,
    layer_num: int,
    contexts: List[str],
    verbose: bool = False,
):
    """Returns the embeddings from transformer models corresponding to the values in ds.

    args:
        ds: A DataSequence containing stimuli for which to retrieve embeddings.
        layer_num: The layer from which to extract embeddings.
        verbose: Print detailed logging information.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n--- Model Setup ---")
        print(f"Device: {device}")
        print(f"Model: {model_name}")
        print(f"Layer: {layer_num}")
        print("-------------------")

    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.output_attentions = False

    model = AutoModel.from_pretrained(model_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    text = np.array(ds.data)
    if verbose:
        print(f"Total input words: {len(text)}")

    new_data = []

    # Build input sequences for all words
    input_sequences = []
    for word_index, word in enumerate(text):
        if contexts is None:
            raise ValueError("Context must be provided.")
        if contexts[word_index] == "":
            context = word
        else:
            context = contexts[word_index] + " " + word
        input_sequences.append(context)

    for word_index, context in enumerate(
        tqdm(input_sequences, desc=f"Generating {model_name} embeddings")
    ):
        if word_index > 50:
            verbose = False  # Only print verbose for first 50 words
        if verbose:
            print(f"\nProcessing word index {word_index}: {text[word_index]}")
            print(f"Context: {context}")

        encoded = tokenizer.encode_plus(
            add_special_tokens=False,
            text=context,
            return_tensors="pt",
        )

        context_only = tokenizer.encode(contexts[word_index], add_special_tokens=False)
        target_word_start_idx = len(context_only)

        tokens_tensor = encoded["input_ids"].to(device)
        if verbose:
            print(
                f"Tokenized context: {tokenizer.convert_ids_to_tokens(tokens_tensor[0])}"
            )
            print(f"Target word start index: {target_word_start_idx}")

        with torch.no_grad():
            outputs = model(tokens_tensor)

            try:
                layer_embedding = outputs.hidden_states[layer_num][0].to("cpu")
            except (AttributeError, IndexError) as e:
                print(
                    f"WARNING: Could not use outputs.hidden_states. Falling back to outputs[-1]. Error: {e}"
                )
                layer_embedding = outputs[-1][layer_num][0].to("cpu")

        word_vector_sequence = layer_embedding[target_word_start_idx:]
        if verbose:
            print(f"Shape of layer embedding: {layer_embedding.shape}")
            print(f"Shape of word vector sequence: {word_vector_sequence.shape}")

        if word_vector_sequence.shape[0] == 0:
            # Fallback if context was truncated or empty
            word_embedding = layer_embedding[-1].numpy()
            if verbose:
                print("Fallback: Using the last token's embedding.")
        else:
            # MEAN average the tokens of the target word
            word_embedding = torch.mean(word_vector_sequence, dim=0).numpy()
            if verbose:
                print(f"Selected word embedding shape: {word_embedding.shape}")

            # new_data.append(layer_embedding[-1].numpy())
        new_data.append(word_embedding)

    if verbose:
        print(f"\n--- Final Data Assembly ---")
        print(f"Total embeddings collected: {len(new_data)}")

    embedding_ds = DataSequence(
        np.array(new_data), ds.split_inds, ds.data_times, ds.tr_times
    )

    if verbose:
        print(f"Final Embedding Data Shape: {embedding_ds.data.shape}")
        print(f"Final Split Indices Length: {len(embedding_ds.split_inds)}")
        print("-----------------------------\n")

    return embedding_ds
