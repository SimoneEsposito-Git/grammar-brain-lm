import os
from typing import Dict, List

import numpy as np
import torch
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
