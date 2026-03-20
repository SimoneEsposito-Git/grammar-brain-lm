import os
from typing import Dict, List

import numpy as np
import torch
import nltk
import spacy
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm


from .data_sequence import DataSequence

def mapdict(d, func):
    """Apply a function to all values in a dictionary."""
    return {k: func(v) for k, v in d.items()}

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
    
def lexical_embeddings(
    dataseqs: dict,
    embedding_path: str,
    downsamp: bool = True,
    pos_tag: list = None
):
    '''Returns lexical embeddings.
    args:
        embedding_path: Path to the embedding file.
        downsample: If True, downsamples responses before returning.
        no_stop_words: bool
            When True, return default embedding for stop words
        pos_tag: str
            When set, return default embedding if tag of the word is not pos_tag
    '''
    english1000_dict = np.load(embedding_path, allow_pickle=True)
    english1000_keys = english1000_dict['keys']
    english1000_values = english1000_dict['values']
    embedding = {k: v for k, v in zip(english1000_keys, english1000_values)}
    default_embedding = np.zeros(english1000_values[0].shape)

    embedding_stimulus = dict()
    for stimulus_name, ds in list(dataseqs.items()):
        print(f'extracting english1000 features for {stimulus_name}')
        embedding_stimulus[stimulus_name] = get_lexical_embeddings(ds=ds, embedding=embedding, default_embedding=default_embedding, pos_tag=pos_tag)
    if downsamp:
        return downsample(embedding_stimulus)
    else:
        return mapdict(embedding_stimulus, lambda h: np.asarray(h.data))
  
def get_lexical_embeddings(
    ds: DataSequence,
    embedding: int,
    default_embedding: np.ndarray,
    pos_tag: list = None
):  
    new_data = []
    text = np.array(ds.data)

    if pos_tag:
        nlp = spacy.load("en_core_web_sm")
        joined = " ".join(text)
        doc = nlp(joined)

        # Build char offsets for each original "word"
        offsets = []
        cursor = 0
        for w in text:
            start = cursor
            end = start + len(str(w))
            offsets.append((start, end))
            cursor = end + 1  # +1 for the space we inserted in join

    for idx, word in enumerate(text):
        tagged = False
        if pos_tag:
            start, end = offsets[idx]
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is not None:
                tagged = any(tok.pos_ in pos_tag for tok in span)
            else:
                tagged = False  # fallback if alignment failed
        print(f'{word} ==> {word if not tagged else "XXXX"}')
        new_data.append(get_word_embedding(word, embedding, default_embedding, tagged))

    embedding_ds = DataSequence(np.array(new_data), ds.split_inds, ds.data_times, ds.tr_times)
    return embedding_ds

def get_word_embedding(
    word: str,
    embedding: Dict,
    default_embedding: np.ndarray,
    pos_tag: bool = False
):
    '''Return the lexical embedding of a given word.

    Parameters:
    ----------
    word : str
        The word for which to get an embedding.
    embedding : dict
        A word:embedding dictionary of word embeddings.
    default_embedding : array_like
        The embedding to use if the word is not in the embedding dictionary.
    no_stop_words: bool
        When True, return default embedding for stop words
    pos_tag: str
        When set, return default embedding if tag of the word is not pos_tag
    '''
    if pos_tag:
        return default_embedding
    try:
        return embedding[word]
    except Exception:
        try:
            lemmatizer = nltk.wordnet.WordNetLemmatizer()
            lemma = lemmatizer.lemmatize(word)
            if pos_tag:
                tag = nltk.pos_tag([lemma], tagset='universal')[0][1]
                if tag != pos_tag:
                    return default_embedding
            return embedding[lemma]
        except:
            print(f'{word} missing from embedding dict.')
            return default_embedding
   
def contextual_embeddings(
    dataseqs: dict,
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
    for story, ds in list(dataseqs.items()):
        stimulus[story] = get_contextual_embeddings(
            ds=ds,
            model_name=model_name,
            layer_num=layer_num,
            contexts=contexts[story],
            verbose=verbose,
            story_name=story,
            model_abbr=model_name.split("/")[-1],
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
    story_name: str = "",
    model_abbr: str = "",
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
        tqdm(input_sequences, desc=f"Generating {model_abbr} embeddings for {story_name}")
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


def masked_embeddings(
    dataseqs: dict,
    model_name: str,
    layer_num: int,
    mask: dict,
    window_size: int = 20,
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
    for story, ds in list(dataseqs.items()):
        stimulus[story] = get_masked_embeddings(
            ds=ds,
            model_name=model_name,
            layer_num=layer_num,
            mask=mask[story],
            window_size=window_size,
            verbose=verbose,
            story_name=story,
            model_abbr=model_name.split("/")[-1],
        )
    if downsamp:
        return downsample(stimulus, interp=interp)
    else:
        return stimulus
    
def get_masked_embeddings(
    ds: DataSequence,
    model_name: str,
    layer_num: int,
    mask: List[bool],
    window_size: int = 20,
    verbose: bool = False,
    story_name: str = "",
    model_abbr: str = "",
):
    """Returns the embeddings from transformer models corresponding to the values in ds.
input_sequence
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
        if mask is None:
            raise ValueError("Mask must be provided.")
        input_sequences.append(text[max(0, word_index - window_size) : word_index + 1])

    for word_index, input_sequence in enumerate(
        tqdm(input_sequences, desc=f"Generating {model_abbr} embeddings for {story_name}")
    ):
        input_sequence_joined = " ".join(input_sequence)
        encoded = tokenizer.encode_plus(
            add_special_tokens=False,
            text=input_sequence_joined,
            return_tensors="pt",
        )
        tokens_tensor = encoded["input_ids"].to(device)
        mask_tensor = words_to_token_mask(
            input_sequence_joined, 
            tokenizer, 
            mask[max(0, word_index - window_size) : word_index + 1]
        )

        try:
            assert tokens_tensor.shape == mask_tensor.shape, "Token tensor and mask tensor must have the same shape."
        except AssertionError as e:
            print(f"Error at word index {word_index} for story {story_name}: {e}")
            print(f"Input sequence: {input_sequence_joined}")
            print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens_tensor[0])}")
            print(f"Token tensor shape: {tokens_tensor.shape}")
            print(f"Mask: {mask_tensor.shape}")
            raise
        if word_index > 50:
            verbose = False  # Only print verbose for first 50 words
        if verbose:
            print(f"\nProcessing word: {input_sequence[-1]} (index {word_index})")
            print(f"Context: {input_sequence_joined}")
            print(f"Mask: {mask[max(0, word_index - window_size) : word_index + 1]}")
            masked_tokens = [tok if m else "[MASK]" for tok, m in zip(tokenizer.convert_ids_to_tokens(tokens_tensor[0]), mask_tensor[0])]
            print(f"Masked tokens: {' '.join(masked_tokens)}")
            #print(f"Tokenized context: {tokenizer.convert_ids_to_tokens(tokens_tensor[0])}")
            #print(f"Token-level mask: {mask_tensor[0].tolist()}")
            #for tok, m in zip(tokenizer.convert_ids_to_tokens(tokens_tensor[0]), mask_tensor[0]):
            #    print(f"Token: {tok}, Masked: {m.item()}")
                
            
        
            
        with torch.no_grad():
            outputs = model(tokens_tensor, attention_mask=mask_tensor.to(device))

            try:
                layer_embedding = outputs.hidden_states[layer_num][0].to("cpu")
            except (AttributeError, IndexError) as e:
                print(
                    f"WARNING: Could not use outputs.hidden_states. Falling back to outputs[-1]. Error: {e}"
                )
                layer_embedding = outputs[-1][layer_num][0].to("cpu")

        
        # average the tokens of the target word that are not masked (mask is False for tokens to keep, True for tokens to mask out)
        masked_embedding = layer_embedding[mask_tensor[0].bool()]
        if masked_embedding.shape[0] == 0:
            # Fallback if all tokens were masked
            word_embedding = layer_embedding[-1].numpy()
            if verbose:
                print("Fallback: Using the last token's embedding.")
        else:
            word_embedding = torch.mean(masked_embedding, dim=0).numpy()
            if verbose:
                print(f"Selected word embedding shape: {word_embedding.shape}")
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

def words_to_token_mask(input_sequence: str, tokenizer: AutoTokenizer, mask: List[bool]) -> torch.Tensor:
    import re
    """Given a list of words and a corresponding mask indicating which words to keep, 
    return a token-level mask that can be applied to the tokenized input."""
    token_mask = []
    words = re.split(r'(?=\s)', input_sequence)
    
    for word, word_mask in zip(words, mask):
        tokens = tokenizer.tokenize(word)
        
        if isinstance(word_mask, list):
            if len(word_mask) != len(tokens):
                try:
                    word_mask = [word_mask[0]] * len(tokens)
                except Exception as e:
                    print(f"Error processing mask for word '{word}' with tokens {tokens}. Mask: {word_mask}")
                    print(f"Sentence: {input_sequence}")
                    raise
            token_mask.extend(word_mask)
        elif isinstance(word_mask, bool):
            token_mask.extend([word_mask] * len(tokens))
        else:
            raise ValueError("Mask must be a list of booleans or a single boolean.")
    
    token_mask[-1] = False
    # flip mask so that True indicates tokens to keep and False indicates tokens to mask out
    token_mask = [not m for m in token_mask]
    return torch.tensor([token_mask]).long()