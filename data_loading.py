import os
from typing import Dict, List

import numpy as np
import h5py
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm

from data_sequence import DataSequence
from textgrid_utils import load_generic_trfiles, load_textgrid_transcripts

# Utility function that may need to be defined or imported from elsewhere
def mapdict(d, func):
    """Apply a function to all values in a dictionary."""
    return {k: func(v) for k, v in d.items()}

# ============================================================================
# Response Processing
# ============================================================================

def load_responses(subjects, modality, split = 'trn', fdir = './'):
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
        fname = os.path.join(fdir, "responses", f"{subject}_{modality}_fmri_data_{split}.hdf")
        with h5py.File(fname) as hf:
            data[subject] = dict()
            for k in hf.keys():
                print("Subject {}, {} will be loaded".format(subject, k))
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

def load_features(split = 'trn', fdir = './'):
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
        for k in hf.keys():
            print("{} will be loaded".format(k))
            data[k] = {}
            for j in hf[k].keys():
                data[k][j] = hf[k][j][()]
    return data

def features_with_embeddings(F, stories, embeddings, name):
    """Merge embedding arrays into feature dictionary for specified stories.
    
    Args:
        F: Feature dictionary to copy and modify.
        stories: List of story identifiers.
        embeddings: Dictionary of embedding arrays by story.
        name: Key name for the embeddings in the feature dictionary.
    
    Returns:
        New feature dictionary with embeddings added.
    """
    new_dict = F.copy()
    for story in stories:
        # Only try to merge if the story exists in the target dictionary
        if story in new_dict.keys():
            new_dict[story][name] = embeddings[story]
    return new_dict

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
        nt,ndim = stim.shape
        dstims = []
        for di,d in enumerate(delays):
            dstim = np.zeros((nt, ndim))
            if d<0: ## negative delay
                dstim[:d,:] = stim[-d:,:]
                if circpad:
                    dstim[d:,:] = stim[:-d,:]
            elif d>0:
                dstim[d:,:] = stim[:-d,:]
                if circpad:
                    dstim[:d,:] = stim[-d:,:]
            else: ## d==0
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

def load_stimulus_word_sequences(stories, names, trfile_dir, transcript_dir):
    """Load word sequences for all stories into DataSequence objects.
    
    Returns:
        Dictionary mapping story identifiers to DataSequence objects.
    """
    wordseq = {}

    trfiles = load_generic_trfiles(names, trfile_dir)
    transcripts = load_textgrid_transcripts(names, transcript_dir)

    for story, name in zip(stories, names):
        ds = DataSequence.from_grid(transcripts[name], trfiles[name])
        wordseq[story] = ds
    
    return wordseq

def downsample(dsdict: Dict, interp: str = 'mean'):
        '''Downsamples each DataSequence in [dsdict] using the settings specified in the
        initializer.
        '''
        # If each value in dict is another dict (e.g. when we get multiple layers of a contextual LM), downsample each dict separately.
        if type(list(dsdict.values())[0]) == dict:
            downsampled_dict = dict()
            for key in dsdict.keys():
                downsampled_dict[key] = mapdict(dsdict[key], lambda h: h.chunksums(interp))
            return downsampled_dict
        else:
            return mapdict(dsdict, lambda h: h.chunksums(interp))

def contextual_embeddings(    wordseqs: dict,
                              model_name: str,
                              layer_num: int,
                              add_special_tokens: bool = True,
                              avg_tokens: bool = True,
                              pretrained: bool = True,
                              context_length: int = 10,
                              downsamp: bool = True,
                              interp: str = 'lanczos'):
        '''Returns embeddings extracted from contextual models.

        Note: Embeddings are currently extracted by feeding in a word and the previous
              (context_length - 1) words. This can be modified by using different context
              methods (eg if sentence markers are given), or by using preceding and subsequent
              words as context.

        args:
            layer_num: The layer from which to extract embeddings.
            model_name: Name of model to use (e.g. 'bert-base-multilingual-cased', 'xlm-mlm-xnli15-1024')
            add_special_tokens: Add the special tokens from the model's tokenizer (e.g. 'CLS' & 'SEP' for BERT)
            avg_tokens: Take the average of the tokens over the window, instead of just the last one.
            pretrained: If True, uses a pretrained model. If False, uses randomly initialized model.
            context_length: The number of words preceeding the embedded word to feed as context.
            downsample: If True, downsamples responses before returning.
        '''
        # Get stimulus for stories.
        stimulus = dict()
        for stimulus_name, ds in list(wordseqs.items()):
            print(f'extracting {model_name} features for {stimulus_name}')
            stimulus[stimulus_name] = get_contextual_embeddings(ds=ds,
                    model_name=model_name,
                    pretrained=pretrained,
                    context_length=context_length,
                    layer_num=layer_num,
                    add_special_tokens=add_special_tokens,
                    avg_tokens=avg_tokens)
        if downsamp:
            return downsample(stimulus, interp=interp)
        else:
            return stimulus
        
def get_contextual_embeddings(ds: DataSequence,
                             model_name: str,
                             context_length: int,
                             layer_num: int,
                             bad_words: List[str] = ['{BR}','{LG}','{LS}','{NS}'],
                             add_special_tokens: bool = True,
                             avg_tokens:bool = True,
                             pretrained: bool = True,
                             verbose: bool = False):
    '''Returns the embeddings from multilingual BERT corresponding to the values in ds.
    
    args:
        ds: A DataSequence containing stimuli for which to retrieve embeddings.
        context_length: The number of words preceeding the embedded word to feed as context.
        layer_num: The layer from which to extract embeddings.
        bad_words: A list of words to ignore.
    '''

    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # --- LOG 1: Device and Model Info (Kept) ---
    if verbose:
        print(f"\n--- Model Setup ---")
        print(f"Target Device: {device}")
        print(f"Model Name: {model_name}")
        print(f"Extraction Layer: {layer_num}")
        print(f"Context Length: {context_length}")
        print("-------------------")
    
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.output_attentions = False
    
    if pretrained:
        model = AutoModel.from_pretrained(model_name, config=config)
    else:
        model = AutoModel.from_config(config)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    
    # --- LOG 2: Total Data Size (Kept) ---
    text = np.array(ds.data)
    print(f"Total input words (text) size: {len(text)}")
    
    new_data = []
    input_sequences = []
    
    for word_index, word in enumerate(text):
        input_sequences.append(text[max(0, word_index - context_length): word_index + 1])
        
    # ----------------------------------------------------------------------
    # 🚀 PROGRESS BAR IMPLEMENTATION 🚀
    # The outer loop is replaced with tqdm to show word processing progress
    # ----------------------------------------------------------------------
    for word_index, input_sequence in enumerate(tqdm(input_sequences, desc=f"Generating {model_name} embeddings")):
        
        # LOG 3, 4, 5, 6 prints are removed or commented out.
        
        input_sequence_bad_words_indices = np.where(np.isin(input_sequence, bad_words))[0]
        input_sequence_cleaned = np.delete(input_sequence, input_sequence_bad_words_indices)
        input_sequence_cleaned = ' '.join(input_sequence_cleaned)
        
        if not input_sequence_cleaned.strip():
            # If the context is only bad words, append a zero-vector placeholder.
            
            # Use config.hidden_size for robustness
            if 'hidden_size' not in config: 
                 # Fallback for models where hidden_size might be nested or named differently
                 hidden_size = 768 
            else:
                hidden_size = config.hidden_size
            
            new_data.append(np.expand_dims(np.zeros(hidden_size), axis=0))
            
            # Optional: Use tqdm.set_postfix to display skipped words if needed
            # tqdm.set_postfix({'status': f'Skipped index {word_index}'})
            continue 
            
        encoded_input_sequence = tokenizer.encode_plus(text=input_sequence_cleaned,
                                                       add_special_tokens=add_special_tokens,
                                                       return_tensors='pt',
                                                       return_special_tokens_mask=add_special_tokens)
        
        tokens_tensor = encoded_input_sequence['input_ids']

        tokens_tensor = tokens_tensor.to(device)
        
        if 'attention_mask' in encoded_input_sequence:
            encoded_input_sequence['attention_mask'] = encoded_input_sequence['attention_mask'].to(device)

        with torch.no_grad():
            
            if 'attention_mask' in encoded_input_sequence:
                outputs = model(input_ids=tokens_tensor, 
                                attention_mask=encoded_input_sequence['attention_mask'])
            else:
                outputs = model(tokens_tensor)
                
            try:
                layer_embedding = outputs.hidden_states[layer_num][0].to('cpu') 
            except (AttributeError, IndexError) as e:
                # Keep the warning print, as this indicates an unexpected model structure
                print(f"WARNING: Could not use outputs.hidden_states. Falling back to outputs[-1]. Error: {e}")
                layer_embedding = outputs[-1][layer_num][0].to('cpu') 
                
            if avg_tokens:
                if add_special_tokens:
                    special_mask_tensor = encoded_input_sequence['special_tokens_mask']
                    special_mask = np.squeeze(special_mask_tensor.numpy())
                    
                    # FIX: Ensure mask is an array and then convert to PyTorch tensor
                    inverted_mask_np = np.atleast_1d(np.logical_not(special_mask))
                    pytorch_mask = torch.from_numpy(inverted_mask_np) 
                    
                    embedding_chunk = layer_embedding[pytorch_mask]
                    new_data.append(np.expand_dims(torch.mean(embedding_chunk, dim=0), axis=0))
                    
                else:
                    new_data.append(np.expand_dims(torch.mean(layer_embedding, dim=0), axis=0))
            else:
                if add_special_tokens:
                    special_mask_tensor = encoded_input_sequence['special_tokens_mask']
                    special_mask = np.squeeze(special_mask_tensor.numpy())
                    
                    # FIX: Ensure mask is an array and then convert to PyTorch tensor
                    inverted_mask_np = np.atleast_1d(np.logical_not(special_mask))
                    pytorch_mask = torch.from_numpy(inverted_mask_np) 
                    
                    new_data.append(np.expand_dims(layer_embedding[pytorch_mask][-1], axis=0))
                else:
                    new_data.append(np.expand_dims(layer_embedding[-1], axis=0))

    # --- LOG 7: Final Data Assembly (Kept) ---
    bad_words_indices = np.where(np.isin(text, bad_words))[0]
    if verbose:
        print(f"\n--- Final Data Assembly ---")
        print(f"Total embeddings collected: {len(new_data)}")
        print(f"Number of bad words removed from tracking: {len(bad_words_indices)}")

    if ds.data_times is not None:
        text_times = ds.data_times
        text_times_cleaned = np.delete(text_times, bad_words_indices)
    else:
        text_times_cleaned = None
        
    split_inds_array = np.array(ds.split_inds)
    
    for index in bad_words_indices[::-1]:
        split_inds_array[split_inds_array > index] = split_inds_array[split_inds_array > index] - 1
        
    embedding_ds = DataSequence(np.squeeze(np.array(new_data)), split_inds_array, text_times_cleaned, ds.tr_times)
    
    # --- LOG 8: Final Output Shape (Kept) ---
    if verbose:
        print(f"Final Embedding Data Shape: {embedding_ds.data.shape}")
        print(f"Final Split Indices Length: {len(embedding_ds.split_inds)}")
        print("-----------------------------\n")
    
    return embedding_ds