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