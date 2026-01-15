import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, NamedTuple
from multiprocessing import Pool, cpu_count
from functools import partial
import os


class MaskedContext(NamedTuple):
    """Structured representation of a context with attention mask information."""
    context: str
    mask_indices: List[int]  # Word-level indices that should be masked


class ContextGenerator:
    def __init__(self, batch_size: int = 64, num_workers: int = None):
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.num_workers = num_workers or min(cpu_count(), 64)

    def _load_model(self):
        """Lazy load GPT-2 model and tokenizer."""
        if self.model is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")
                # Enable optimizations for H100
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def generate_context(
        self,
        ds,
        mode: str,
        window_size: int = 10,
        amount: int = 3,
        seed: int = 42,
        story: str = "",
    ) -> List[MaskedContext]:
        """Generate contexts based on the specified mode."""
        if mode == "random":
            return self._generate_random_context(ds, window_size, seed, story)
        elif mode == "baseline":
            return self._generate_baseline_context(ds, window_size, story)
        elif mode in ["peak", "valley"]:
            return self._generate_entropy_masks(ds, window_size, amount, mode, story)
        elif mode in ["remove-peak", "remove-valley"]:
            return self._generate_entropy_masks(
                ds, window_size, amount, mode[7:], story, remove=True
            )
        elif mode == "zero":
            return [MaskedContext("", []) for _ in range(len(ds.data))]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_baseline_context(
        self, ds, window_size: int = 10, story: str = ""
    ) -> List[MaskedContext]:
        """Generate baseline contexts with preceding words (parallelized)."""
        text = np.array(ds.data)

        # Parallel context extraction
        with Pool(self.num_workers) as pool:
            all_contexts = list(
                tqdm(
                    pool.imap(
                        partial(
                            _extract_baseline_context,
                            text=text,
                            window_size=window_size,
                        ),
                        range(len(text)),
                    ),
                    total=len(text),
                    desc=f"Generating baseline contexts for story: {story}",
                )
            )

        return all_contexts

    def _calculate_surprisals_batch(
        self, context_words_list: List[List[str]]
    ) -> List[np.ndarray]:
        """Calculate surprisals for multiple contexts (non-batched for clarity)."""
        all_surprisals = []

        for context_words in context_words_list:
            if not context_words:
                all_surprisals.append(np.array([]))
                continue

            word_surprisals = []
            
            # Calculate surprisal for each word in context
            for i, word in enumerate(context_words):
                prefix = " ".join(context_words[:i])
                target = word

                if not target or not target.strip():
                    word_surprisals.append(0.0)
                    continue

                # Get token IDs for prefix and target
                target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
                if not target_tokens:
                    word_surprisals.append(0.0)
                    continue

                # Calculate surprisal for each token in the target word
                word_surprisal = 0.0
                for token_idx, target_id in enumerate(target_tokens):
                    # Build context for this token
                    if token_idx == 0:
                        context_ids = self.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
                    else:
                        # Include previously decoded tokens from target
                        prev_tokens = target_tokens[:token_idx]
                        prev_text = self.tokenizer.decode(prev_tokens).strip()
                        full_prefix = (prefix + " " + prev_text).strip() if prefix else prev_text
                        context_ids = self.tokenizer.encode(full_prefix, add_special_tokens=False)

                    # Get model prediction
                    if context_ids:
                        input_tensor = torch.tensor([context_ids]).to(self.model.device)
                    else:
                        # Unconditional: use empty input (model will use implicit BOS)
                        input_tensor = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        probs = torch.softmax(logits, dim=-1)
                        prob = probs[target_id].item()
                        surprisal = -np.log2(prob + 1e-10)
                        word_surprisal += surprisal

                # Average surprisal across tokens in word
                avg_surprisal = word_surprisal / len(target_tokens)
                word_surprisals.append(avg_surprisal)

            surprisals_array = np.array(word_surprisals)
            all_surprisals.append(surprisals_array)
            
            # DEBUG: Print surprisals for verification
            print(f"DEBUG surprisals: context={context_words}, surprisals={surprisals_array}")

        return all_surprisals

    def _get_mask_indices(
        self, word_surprisals: np.ndarray, amount: int, mode: str
    ) -> np.ndarray:
        """Get indices to mask based on surprisal values."""
        if amount <= 0 or len(word_surprisals) == 0:
            return np.array([], dtype=int)

        amount = min(amount, len(word_surprisals))
        if mode == "peak":
            return word_surprisals.argsort()[-amount:]
        else:
            return word_surprisals.argsort()[:amount]

    def _generate_entropy_masks(
        self,
        ds,
        window_size: int = 10,
        amount: int = 3,
        mode: str = "peak",
        story: str = "",
        remove: bool = False,
    ) -> List[MaskedContext]:
        """Generate contexts with entropy-based masking (optimized with batching)."""
        self._load_model()

        text = np.array(ds.data)

        # Extract all contexts in parallel
        with Pool(self.num_workers) as pool:
            context_words_list = list(
                pool.imap(
                    partial(_extract_context_words, text=text, window_size=window_size),
                    range(len(text)),
                )
            )

        # Calculate surprisals in batches
        all_surprisals = []
        for i in tqdm(
            range(0, len(context_words_list), self.batch_size),
            desc=f"Generating {mode} contexts for story: {story}",
        ):
            batch_contexts = context_words_list[i : i + self.batch_size]
            batch_surprisals = self._calculate_surprisals_batch(batch_contexts)
            all_surprisals.extend(batch_surprisals)

        # Apply masking in parallel
        with Pool(self.num_workers) as pool:
            all_contexts = list(
                pool.starmap(
                    (
                        partial(_remove_words, amount=amount, mode=mode)
                        if remove
                        else partial(_apply_masking, amount=amount, mode=mode)
                    ),
                    zip(context_words_list, all_surprisals),
                )
            )

        return all_contexts

    def _generate_random_context(
        self, ds, window_size: int = 10, seed: int = 42, story: str = ""
    ) -> List[MaskedContext]:
        """Generate randomized contexts for null hypothesis testing (parallelized)."""
        text = np.array(ds.data)

        # Parallel random shuffling
        with Pool(self.num_workers) as pool:
            all_contexts = list(
                tqdm(
                    pool.imap(
                        partial(
                            _shuffle_context,
                            text=text,
                            window_size=window_size,
                            seed=seed,
                        ),
                        range(len(text)),
                    ),
                    total=len(text),
                    desc=f"Generating Random Context (Null Hypothesis) for: {story}",
                )
            )

        return all_contexts


# Helper functions for multiprocessing (must be at module level)
def _extract_baseline_context(i: int, text: np.ndarray, window_size: int) -> MaskedContext:
    """Extract baseline context for a single position."""
    context_str = " ".join(text[max(0, i - window_size) : i])
    return MaskedContext(context_str, [])


def _extract_context_words(i: int, text: np.ndarray, window_size: int) -> List[str]:
    """Extract context words for a single position."""
    start_idx = max(0, i - window_size)
    return text[start_idx:i].tolist()


def _apply_masking(
    context_words: List[str], word_surprisals: np.ndarray, amount: int, mode: str
) -> MaskedContext:
    """Apply masking to context based on surprisal values."""
    if not context_words or len(word_surprisals) == 0:
        return MaskedContext(" ".join(context_words), [])

    amount = min(amount, len(word_surprisals))
    if amount <= 0:
        return MaskedContext(" ".join(context_words), [])

    if mode == "peak":
        mask_indices = word_surprisals.argsort()[-amount:].tolist()
    else:
        mask_indices = word_surprisals.argsort()[:amount].tolist()

    return MaskedContext(" ".join(context_words), mask_indices)

"""
def _remove_words(
    context_words: List[str], word_surprisals: np.ndarray, amount: int, mode: str
) -> MaskedContext:
    ""Remove words from context based on surprisal values, returning mask indices.""
    if not context_words or len(word_surprisals) == 0:
        return MaskedContext("", [])

    amount = min(amount, len(word_surprisals))
    if amount <= 0:
        return MaskedContext(" ".join(context_words), [])

    if mode == "peak":
        remove_indices = word_surprisals.argsort()[-amount:].tolist()
    else:
        remove_indices = word_surprisals.argsort()[:amount].tolist()

    remaining_words = [
        word for idx, word in enumerate(context_words) if idx not in remove_indices
    ]

    return MaskedContext(" ".join(remaining_words), remove_indices)
"""

def _shuffle_context(i: int, text: np.ndarray, window_size: int, seed: int) -> MaskedContext:
    """Shuffle context for a single position."""
    # Create position-specific seed for reproducibility
    rng = random.Random(seed + i)

    start_idx = max(0, i - window_size)
    context_words = text[start_idx:i].tolist()

    if not context_words:
        return MaskedContext("", [])

    rng.shuffle(context_words)
    return MaskedContext(" ".join(context_words), [])
