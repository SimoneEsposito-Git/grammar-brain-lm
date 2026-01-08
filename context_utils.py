import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import os
import data_loading as dl
import numpy as np
from tqdm import tqdm

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Load spaCy for POS tagging
nlp = spacy.load("en_core_web_sm")

"""
def get_word_entropy(context_text, target_word):
    inputs = tokenizer(context_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits[0, -1, :]  # Logits for the next word
        probabilities = torch.softmax(predictions, dim=-1)
    # Get ID of the target word
    target_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    prob = probabilities[target_id].item()
    return -torch.log2(torch.tensor(prob)).item()

def find_entropy_match(context_text, original_word, tolerance=0.1):
    original_entropy = get_word_entropy(context_text, original_word)
    doc = nlp(context_text+original_word)
    original_pos = doc[-1].pos_
    inputs = tokenizer(context_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    top_values, top_indices = torch.topk(probs, 500)
    candidates = []
    for i in range(len(top_indices)):
        word = tokenizer.decode([top_indices[i]]).strip()
        prob = top_values[i].item()
        candidate_entropy = -torch.log2(torch.tensor(prob)).item()
        
        # print(word, candidate_entropy)
        # Check POS and Entropy distance
        
        if abs(candidate_entropy - original_entropy) < tolerance:
            # Linguistic constraint: check POS
            doc = nlp(word)
            # print(original_pos, doc[0].pos_)
            if doc and doc[0].pos_ == original_pos and word.strip().lower() != original_word.strip().lower():
                candidates.append((word, candidate_entropy))
                
    # Return the closest match
    return sorted(candidates, key=lambda x: abs(x[1] - original_entropy))

#print(find_entropy_match("is not the only universe there is. There are", " alternate"))

def data_to_entropy_map(ds, context_length):
    text = np.array(ds.data)
    new_data = []
    input_sequences = []
    
    for word_index, word in enumerate(text):
        input_sequence = text[max(0, word_index - context_length): word_index + 1]
        input_sequences.append(input_sequence)
        
    for seq in input_sequences:
        if len(seq) < 2:
            new_data.append(seq[-1].item())
            continue
        context = " ".join(seq[:-1]) 
        target = seq[-1]
        # print("Context:", repr(context), "Target:", repr(target))
        entropy_match = find_entropy_match(context, " " + target)
        if len(entropy_match) > 0:
            # print("Match found:", entropy_match[0])
            new_data.append(entropy_match[0][0])
            print(target.item(), "->", entropy_match[0][0])
        else:
            new_data.append(target.item())
            print("No match for:", repr(target.item()))
    return new_data 
"""

def generate_context(ds, mode, **kwargs):
    if mode == "random":
        return precompute_random_masks(ds, **kwargs)
    elif mode in ["peak", "valley"]:
        return precompute_entropy_masks(ds, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def precompute_entropy_masks(ds, window_size=10, amount=3, mode="peak"):
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
    for i in tqdm(range(len(text)), desc=f"Masking {mode}s"):
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
