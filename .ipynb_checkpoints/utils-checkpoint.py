"""
Utility function: loading data from hdf5 files and loading mapper files to display data on the
cortical surface.

"""

import numpy as np
import itertools as itools
import scipy.sparse
import h5py
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group
import torch
from data_sequence import DataSequence 
from typing import Dict, List, Union
try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except Exception:
    print('Could not import BertModel, BertTokenizer from transformers')
    
def load_data(fname, key=None):
    """Function to load data from an hdf file.

    Parameters
    ----------
    fname: string
        hdf5 file name
    key: string
        key name to load. If not provided, all keys will be loaded.

    Returns
    -------
    data : dictionary
        dictionary of arrays

    """
    data = dict()
    with h5py.File(fname) as hf:
        if key is None:
            for k in hf.keys():
                print("{} will be loaded".format(k))
                if type(hf[k]) == Dataset:
                    data[k] = hf[k][()]
                if type(hf[k]) == Group:
                    data[k] = {}
                    for j in hf[k].keys():
                        data[k][j] = hf[k][j][()]
        else:
            data[key] = hf[key][()]
    return data


def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

    By Mark Lescroart

    """
    with h5py.File(fname) as hf:
        data = (hf['%s_data'%varname], hf['%s_indices'%varname], hf['%s_indptr'%varname])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape'%varname])
    return sparsemat


def map_to_flat(voxels, mapper_file):
    """Generate flatmap image for an individual subject from voxel array

    This function maps a list of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array
        n x 1 array of voxel values to be mapped
    mapper_file: string
        file containing mapping arrays

    Returns
    -------
    image : array
        flatmap image, (n x 1024)

    By Mark Lescroart

    """
    pixmap = load_sparse_array(mapper_file, 'voxel_to_flatmap')
    with h5py.File(mapper_file, mode='r') as hf:
        pixmask = hf['flatmap_mask'][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]

"""
def get_contextual_embeddings(text, tokenizer, model, args,
        bad_words=['paragraph_start', 'paragraph_end', 'clause_start', 'clause_end', 'sentence_start', 'sentence_end'],
        model_start_delimiter='[CLS]',
        model_end_delimiter='[SEP]',
        layer_nums=range(13),
        default_context_length=10,):
    '''
    Arguments:
        text: list: List of input words.
    Returns:
        embeddings_dict_bylayer: dict: Dictionary of layer_num:[num_words x embedding_size] numpy array of contextual embeddings.
    '''
    model.eval()

    newdata = []
    
    inputsequences = []
    if args.context_length < 1:
        inputsequence_starts = np.where(np.array(text) == args.sequence_start_delimiter)[0]
        inputsequence_ends = np.where(np.array(text) == args.sequence_end_delimiter)[0]
        for start, end in zip(inputsequence_starts, inputsequence_ends):
            inputsequences.append(text[start:end+1])
    else:
        for word_index, word in enumerate(text):
            inputsequences.append(text[max(0, word_index - args.context_length):word_index+1])
    
    context_length = args.context_length
    if len(inputsequences) < 1:
        context_length = default_context_length
        warnings.warn('Using default context length {default_context_length}.'.format(default_context_length=default_context_length))
        for word_index, word in enumerate(text):
            inputsequences.append(text[max(0, word_index - default_context_length):word_index+1])

    inputsequences_cleaned = []
    
    embeddings_dict_bylayer = dict()
    for layer_num in layer_nums:
        embeddings_dict_bylayer['layer{layer_num}'.format(layer_num=layer_num)] = []
   
    for inputsequence in inputsequences:
        # take out paragraph_start, paragraph_end, clause_start, clause_end.
        inputsequence_bad_words_indices = np.where(np.isin(inputsequence, bad_words))[0]
        inputsequence_cleaned = np.delete(inputsequence, inputsequence_bad_words_indices)

        # Get indices of first tokens in words.
        embedding_index = 0
        word_end_indices = []
        for word in inputsequence_cleaned:
            word_length = len(tokenizer.tokenize(word))
            word_end_indices.append(embedding_index + word_length - 1)
            embedding_index += word_length

        # Get embeddings.
        inputsequence_cleaned = ' '.join(inputsequence_cleaned)
        if args.add_mBERT_delimiters:
            inputsequence_cleaned = model_start_delimiter + ' ' + inputsequence_cleaned + ' ' + model_end_delimiter

        tokenized_inputsequence = tokenizer.tokenize(inputsequence_cleaned)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_inputsequence)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = model(tokens_tensor)
            for layer_num in layer_nums:
                encoded_layers = outputs[2][layer_num][0]
                if context_length < 1:
                   embeddings_dict_bylayer['layer{layer_num}'.format(layer_num=layer_num)].append(np.atleast_2d(np.squeeze(encoded_layers.numpy()[word_end_indices])))
                else:
                    embeddings_dict_bylayer['layer{layer_num}'.format(layer_num=layer_num)].append(np.atleast_2d(np.squeeze(encoded_layers.numpy()[-1])))
 
    for layer_num in layer_nums:
        embeddings_dict_bylayer['layer{layer_num}'.format(layer_num=layer_num)] = np.concatenate(embeddings_dict_bylayer['layer{layer_num}'.format(layer_num=layer_num)], axis=0)

    return embeddings_dict_bylayer, context_length
"""
def get_chunk_indices(N, num_chunks):
    """
    Calculates the start indices for splitting a list of length N 
    into a specified number of approximately equal-sized chunks (num_chunks).

    Args:
        N (int): The total length of the list.
        num_chunks (int): The number of chunks (N') desired.

    Returns:
        list: A list of indices indicating where each chunk starts.
    """
    if num_chunks <= 0:
        return []
    if N <= 0:
        return [0]

    # The minimum number of items per chunk (integer division)
    base_size = N // num_chunks
    # The number of chunks that will have one extra item (the remainder)
    remainder = N % num_chunks

    indices = [0]  # The first chunk always starts at index 0
    current_index = 0

    for i in range(num_chunks):
        # Determine the size of the current chunk
        # Add 1 for the first 'remainder' chunks
        chunk_size = base_size + (1 if i < remainder else 0)

        # The end index of the current chunk is the start index of the next
        current_index += chunk_size

        # Only append the index if it's not the very last index (N), 
        # as we are defining the *start* of the next chunk.
        # We append the final index (N) to define the end of the last chunk.
        if i < num_chunks - 1:
            indices.append(current_index)
        else:
            # Append the final index (N) to define the end boundary for slicing
            indices.append(N) 

    # We only need the starting indices for the N' chunks, 
    # and the final index N to end the last slice.
    return indices[1:-1]
    
def downsample(dsdict: Dict, interp: str = 'lanczos'):
        '''Downsamples each DataSequence in [dsdict] using the settings specified in the
        initializer.
        '''
        # If each value in dict is another dict (e.g. when we get multiple layers of a contextual LM), downsample each dict separately.
        if type(list(dsdict.values())[0]) == dict:
            downsampled_dict = dict()
            for key in dsdict.keys():
                downsampled_dict[key] = mapdict(dsdict[key], lambda h: h.chunksums(interp, **interpargs))
            return downsampled_dict
        else:
            return mapdict(dsdict, lambda h: h.chunksums(interp, **interpargs))

def contextual_embeddings(    wordseqs: dict,
                              model_name: str,
                              layer_num: int,
                              add_special_tokens: bool = True,
                              avg_tokens: bool = True,
                              pretrained: bool = True,
                              context_length: int = 10,
                              downsample: bool = True):
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
        if downsample:
            return downsample(stimulus)
        else:
            return stimulus
        
def get_contextual_embeddings(ds: DataSequence,
                             model_name: str,
                             context_length: int,
                             layer_num: int,
                             bad_words: List[str] = ['{BR}','{LG}','{LS}','{NS}'],
                             add_special_tokens: bool = True,
                             avg_tokens:bool = True,
                             pretrained: bool = True):
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

    # --- LOG 1: Device and Model Info ---
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
    
    # --- LOG 2: Total Data Size ---
    text = np.array(ds.data)
    print(f"Total input words (text) size: {len(text)}")
    print(f"Number of chunks (input_sequences) to process: {len(text)}")

    new_data = []
    input_sequences = []
    
    for word_index, word in enumerate(text):
        input_sequences.append(text[max(0, word_index - context_length): word_index + 1])
        
    for word_index, input_sequence in enumerate(input_sequences):
        
        # --- LOG 3: Per-word/Sequence Debugging (every N words) ---
        if word_index % 50 == 0 or word_index == len(input_sequences) - 1:
            print(f"\nProcessing word index {word_index}/{len(text)-1}...")
            print(f"  Input sequence (raw, length {len(input_sequence)}): {' '.join(input_sequence[-3:])}...") # Show last few words
        
        input_sequence_bad_words_indices = np.where(np.isin(input_sequence, bad_words))[0]
        input_sequence_cleaned = np.delete(input_sequence, input_sequence_bad_words_indices)
        input_sequence_cleaned = ' '.join(input_sequence_cleaned)
        
        encoded_input_sequence = tokenizer.encode_plus(text=input_sequence_cleaned,
                                                       add_special_tokens=add_special_tokens,
                                                       return_tensors='pt',
                                                       return_special_tokens_mask=add_special_tokens)
        
        tokens_tensor = encoded_input_sequence['input_ids']

        # Move the input tensors to the same device as the model
        tokens_tensor = tokens_tensor.to(device)
        
        # --- LOG 4: Tokenization and Device Check ---
        if word_index % 50 == 0 or word_index == len(input_sequences) - 1:
            print(f"  Cleaned input (for tokenizer): \"{input_sequence_cleaned}\"")
            print(f"  Input Tokens Tensor shape: {tokens_tensor.shape}, Device: {tokens_tensor.device}")
            
        # You should also move the attention_mask if it exists
        if 'attention_mask' in encoded_input_sequence:
            encoded_input_sequence['attention_mask'] = encoded_input_sequence['attention_mask'].to(device)

        with torch.no_grad():
            
            # Use attention mask if available (Best practice)
            if 'attention_mask' in encoded_input_sequence:
                outputs = model(input_ids=tokens_tensor, 
                                attention_mask=encoded_input_sequence['attention_mask'])
            else:
                outputs = model(tokens_tensor)
                
            # Accessing hidden states using outputs.hidden_states is safer
            try:
                # Assuming layer_embedding is extracted from outputs.hidden_states
                layer_embedding = outputs.hidden_states[layer_num][0].to('cpu') 
            except (AttributeError, IndexError) as e:
                # Fallback if outputs structure is different, using original logic
                print(f"WARNING: Could not use outputs.hidden_states. Falling back to outputs[-1]. Error: {e}")
                layer_embedding = outputs[-1][layer_num][0].to('cpu') 
                
            # --- LOG 5: Embedding Shape and Device Check ---
            if word_index % 50 == 0 or word_index == len(input_sequences) - 1:
                print(f"  Layer Embedding Shape (on CPU): {layer_embedding.shape}") # [num_tokens x hidden_size]
            
            if avg_tokens:
                if add_special_tokens:
                    # 1. Access the special_tokens_mask tensor
                    special_mask_tensor = encoded_input_sequence['special_tokens_mask']
                    
                    # 2. Convert to NumPy and SQUEEZE to ensure it's 1-dimensional
                    # This is the fix for the IndexError: shape mismatch
                    special_mask = np.squeeze(special_mask_tensor.numpy())
                    
                    # --- LOG 6: Mask Shape Check ---
                    if word_index % 50 == 0 or word_index == len(input_sequences) - 1:
                        print(f"  Special Mask shape (NumPy): {special_mask.shape}, Sum (tokens to discard): {np.sum(special_mask)}")

                    # Apply boolean mask
                    embedding_chunk = layer_embedding[np.logical_not(special_mask)]
                    new_data.append(np.expand_dims(torch.mean(embedding_chunk, dim=0), axis=0))
                    
                else:
                    new_data.append(np.expand_dims(torch.mean(layer_embedding, dim=0), axis=0))
            else:
                if add_special_tokens:
                    # Take the last token after discarding the special ones so that it corresponds to the last word
                    
                    # Fix special_mask creation here too, if avg_tokens is False
                    special_mask_tensor = encoded_input_sequence['special_tokens_mask']
                    special_mask = np.squeeze(special_mask_tensor.numpy())
                    
                    new_data.append(np.expand_dims(layer_embedding[np.logical_not(special_mask)][-1], axis=0))
                else:
                    new_data.append(np.expand_dims(layer_embedding[-1], axis=0))

    # --- LOG 7: Final Data Assembly ---
    bad_words_indices = np.where(np.isin(text, bad_words))[0]
    print(f"\n--- Final Data Assembly ---")
    print(f"Total embeddings collected: {len(new_data)}")
    print(f"Number of bad words removed from tracking: {len(bad_words_indices)}")

    text_times = ds.data_times
    text_times_cleaned = np.delete(text_times, bad_words_indices)
    split_inds_array = np.array(ds.split_inds)
    
    # ... (split index adjustment logic) ...
    
    embedding_ds = DataSequence(np.squeeze(np.array(new_data)), split_inds_array, text_times_cleaned, ds.tr_times)
    
    # --- LOG 8: Final Output Shape ---
    print(f"Final Embedding Data Shape: {embedding_ds.data.shape}")
    print(f"Final Split Indices Length: {len(embedding_ds.split_inds)}")
    print("-----------------------------\n")
    
    return embedding_ds