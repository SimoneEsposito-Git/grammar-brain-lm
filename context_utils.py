import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import os
import data_loading as dl
import numpy as np
from tqdm import tqdm
import random

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def generate_context(ds, mode, **kwargs):
    if mode == "random":
        return precompute_random_context(ds, **kwargs)
    elif mode == "baseline":
        return precompute_baseline_context(ds, **kwargs)
    elif mode in ["peak", "valley"]:
        return precompute_entropy_masks(ds, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def precompute_baseline_context(ds, window_size=10, story=""):
    """
    Args:
        ds: The DataSequence object.
        window_size: Number of words preceding the target word.
    Returns:
        List[str]: List of context strings.
    """

    text = np.array(ds.data)
    all_contexts = []
    for word_index, word in tqdm(
        enumerate(text), desc=f"Generating baseline contexts for story: {story}"
    ):
        all_contexts.append(
            " ".join(text[max(0, word_index - window_size) : word_index + 1])
        )
    return all_contexts


def precompute_entropy_masks(ds, window_size=10, amount=3, mode="peak", story=""):
    """
    Args:
        ds: The DataSequence object.
        window_size: Number of words preceding the target word.
        amount: How many words to mask (e.g., 3).
        mode: "peak" to mask highest entropy, "valley" for lowest.
    Returns:
        List[str]: List of masked context strings.
    """
    # 1. Setup Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    text = np.array(ds.data)
    all_contexts = []

    # We loop through every word in the stimulus
    for i in tqdm(
        range(len(text)), desc=f"Generating {mode} contexts for story: {story}"
    ):
        # Get the context window (excluding the word at index i)
        start_idx = max(0, i - window_size)
        context_words = text[start_idx:i].tolist()

        if not context_words:
            all_contexts.append("")
            continue

        # 2. Calculate Surprisal for each word in the window
        # To get accurate entropy, we need the context preceding EACH word in the window
        word_surprisals = []
        for j in range(len(context_words)):
            # Segment of text before the word we are currently measuring
            prefix = " ".join(text[start_idx : start_idx + j])
            target = context_words[j]

            # Check if target is valid and can be encoded
            if not target or not target.strip():
                word_surprisals.append(0.0)  # Default surprisal for empty targets
                continue

            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            if not target_tokens:
                word_surprisals.append(0.0)  # Default surprisal for unencodable targets
                continue

            # Simple entropy calculation
            inputs = tokenizer(prefix if prefix else target, return_tensors="pt").to(
                model.device
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

            target_id = target_tokens[0]
            p_word = probs[target_id].item()
            surprisal = -np.log2(p_word + 1e-10)  # 1e-10 prevents log(0)
            word_surprisals.append(surprisal)

        # 3. Identify Indices to Mask
        # Get indices of the highest (peak) or lowest (valley) values
        word_surprisals = np.array(word_surprisals)
        if mode == "peak":
            # Indices of the largest values
            mask_indices = word_surprisals.argsort()[-amount:]
        else:
            # Indices of the smallest values
            mask_indices = word_surprisals.argsort()[:amount]

        # 4. Apply Mask
        masked_words = context_words.copy()
        for idx in mask_indices:
            masked_words[idx] = "[MASK]"

        all_contexts.append(" ".join(masked_words))

    return all_contexts


def precompute_random_context(ds, window_size=10, seed=42, story=""):
    """
    Args:
        ds: The DataSequence object containing the story words.
        window_size: The number of words preceding the target.
        seed: Random seed for reproducibility (vital for science!).
    Returns:
        List[str]: A list of strings where each string is a shuffled context.
    """
    # Set seed to ensure the "randomness" is the same every time you run it
    random.seed(seed)
    np.random.seed(seed)

    text = np.array(ds.data)
    all_contexts = []

    for i in tqdm(range(len(text)), desc="Generating Random Context (Null Hypothesis) for: {story}"):
        # 1. Get the context window (excluding the target word at index i)
        start_idx = max(0, i - window_size)
        context_words = text[start_idx:i].tolist()

        if not context_words:
            all_contexts.append("")
            continue

        # 2. Shuffle the words (Destroys syntax and narrative)
        # .shuffle() happens in-place
        random.shuffle(context_words)

        # 3. Join and store
        all_contexts.append(" ".join(context_words))

    return all_contexts
