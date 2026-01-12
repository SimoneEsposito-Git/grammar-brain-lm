import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial
import os


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
    ) -> List[str]:
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
            return [""] * len(ds.data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_baseline_context(
        self, ds, window_size: int = 10, story: str = ""
    ) -> List[str]:
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
        """Calculate surprisals for multiple contexts in batches."""
        all_surprisals = []

        for context_words in context_words_list:
            if not context_words:
                all_surprisals.append(np.array([]))
                continue

            word_surprisals = []
            batch_prefixes = []
            batch_targets = []
            batch_indices = []

            # Prepare all prefix-target pairs
            for j in range(len(context_words)):
                prefix = " ".join(context_words[:j])
                target = context_words[j]

                if not target or not target.strip():
                    word_surprisals.append(0.0)
                    continue

                target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
                if not target_tokens:
                    word_surprisals.append(0.0)
                    continue

                # Create prefix-target pairs for each token
                for token_idx, target_id in enumerate(target_tokens):
                    if token_idx == 0:
                        temp_prefix = prefix
                    else:
                        prev = self.tokenizer.decode(target_tokens[:token_idx]).strip()
                        temp_prefix = (prefix + " " + prev).strip() if prefix else prev

                    batch_prefixes.append(temp_prefix if temp_prefix.strip() else None)
                    batch_targets.append(target_id)
                    batch_indices.append((j, len(target_tokens)))

            # Batch process all prefixes
            if batch_prefixes:
                token_surprisals = self._batch_calculate_token_surprisals(
                    batch_prefixes, batch_targets
                )

                # Aggregate token surprisals into word surprisals
                current_word_idx = -1
                current_word_surprisal = 0.0
                current_token_count = 0

                for (word_idx, total_tokens), surprisal in zip(
                    batch_indices, token_surprisals
                ):
                    if word_idx != current_word_idx:
                        if current_word_idx >= 0:
                            word_surprisals.append(
                                current_word_surprisal / current_token_count
                            )
                        current_word_idx = word_idx
                        current_word_surprisal = surprisal
                        current_token_count = 1
                    else:
                        current_word_surprisal += surprisal
                        current_token_count += 1

                if current_word_idx >= 0:
                    word_surprisals.append(current_word_surprisal / current_token_count)

            all_surprisals.append(np.array(word_surprisals))

        return all_surprisals

    def _batch_calculate_token_surprisals(
        self, prefixes: List[str], target_ids: List[int]
    ) -> List[float]:
        """Calculate surprisals for tokens in batches using GPU."""
        surprisals = []

        # Cache unconditional logits (calculated once)
        unconditional_probs = None

        for i in range(0, len(prefixes), self.batch_size):
            batch_prefixes = prefixes[i : i + self.batch_size]
            batch_targets = target_ids[i : i + self.batch_size]

            # Handle None prefixes (unconditional probability)
            valid_mask = [p is not None for p in batch_prefixes]
            valid_prefixes = [p for p in batch_prefixes if p is not None]

            if valid_prefixes:
                # Tokenize batch
                inputs = self.tokenizer(
                    valid_prefixes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Get last token logits for each sequence
                    last_token_logits = outputs.logits[
                        range(len(valid_prefixes)), inputs.attention_mask.sum(dim=1) - 1
                    ]
                    probs = torch.softmax(last_token_logits, dim=-1)

                # Extract probabilities for target tokens
                valid_idx = 0
                for j, is_valid in enumerate(valid_mask):
                    if is_valid:
                        p_token = probs[valid_idx, batch_targets[j]].item()
                        surprisals.append(-np.log2(p_token + 1e-10))
                        valid_idx += 1
                    else:
                        # Lazy compute unconditional probability
                        if unconditional_probs is None:
                            # Use BOS token as context
                            bos_input = torch.tensor(
                                [[self.tokenizer.bos_token_id]],
                                device=self.model.device,
                            )
                            with torch.no_grad():
                                outputs_uncond = self.model(input_ids=bos_input)
                                unconditional_probs = torch.softmax(
                                    outputs_uncond.logits[0, -1, :], dim=-1
                                )

                        p_token = unconditional_probs[batch_targets[j]].item()
                        surprisals.append(-np.log2(p_token + 1e-10))
            else:
                # All unconditional - compute once and reuse
                if unconditional_probs is None:
                    bos_input = torch.tensor(
                        [[self.tokenizer.bos_token_id]], device=self.model.device
                    )
                    with torch.no_grad():
                        outputs_uncond = self.model(input_ids=bos_input)
                        unconditional_probs = torch.softmax(
                            outputs_uncond.logits[0, -1, :], dim=-1
                        )

                for target_id in batch_targets:
                    p_token = unconditional_probs[target_id].item()
                    surprisals.append(-np.log2(p_token + 1e-10))

        return surprisals

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
    ) -> List[str]:
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
    ) -> List[str]:
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
def _extract_baseline_context(i: int, text: np.ndarray, window_size: int) -> str:
    """Extract baseline context for a single position."""
    return " ".join(text[max(0, i - window_size) : i])


def _extract_context_words(i: int, text: np.ndarray, window_size: int) -> List[str]:
    """Extract context words for a single position."""
    start_idx = max(0, i - window_size)
    return text[start_idx:i].tolist()


def _apply_masking(
    context_words: List[str], word_surprisals: np.ndarray, amount: int, mode: str
) -> str:
    """Apply masking to context based on surprisal values."""
    if not context_words or len(word_surprisals) == 0:
        return ""

    amount = min(amount, len(word_surprisals))
    if amount <= 0:
        return " ".join(context_words)

    if mode == "peak":
        mask_indices = word_surprisals.argsort()[-amount:]
    else:
        mask_indices = word_surprisals.argsort()[:amount]

    masked_words = context_words.copy()
    for idx in mask_indices:
        if 0 <= idx < len(masked_words):
            masked_words[idx] = "[MASK]"

    return " ".join(masked_words)


def _remove_words(
    context_words: List[str], word_surprisals: np.ndarray, amount: int, mode: str
) -> str:
    """Remove words from context based on surprisal values."""
    if not context_words or len(word_surprisals) == 0:
        return ""

    amount = min(amount, len(word_surprisals))
    if amount <= 0:
        return " ".join(context_words)

    if mode == "peak":
        remove_indices = word_surprisals.argsort()[-amount:]
    else:
        remove_indices = word_surprisals.argsort()[:amount]

    remaining_words = [
        word for idx, word in enumerate(context_words) if idx not in remove_indices
    ]

    return " ".join(remaining_words)


def _shuffle_context(i: int, text: np.ndarray, window_size: int, seed: int) -> str:
    """Shuffle context for a single position."""
    # Create position-specific seed for reproducibility
    rng = random.Random(seed + i)

    start_idx = max(0, i - window_size)
    context_words = text[start_idx:i].tolist()

    if not context_words:
        return ""

    rng.shuffle(context_words)
    return " ".join(context_words)
