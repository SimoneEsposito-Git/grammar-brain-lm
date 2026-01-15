import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List, NamedTuple

class MaskedContext(NamedTuple):
    """Structured representation of a context with attention mask information."""
    context: str
    mask_indices: List[int]  # Word-level indices that should be masked
    
class ContextGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Lazy load GPT-2 model and tokenizer."""
        if self.model is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")

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
        elif mode == "zero":
            return [MaskedContext(context="", mask_indices=[]) for _ in range(len(ds.data))]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_baseline_context(
        self, ds, window_size: int = 10, story: str = ""
    ) -> List[MaskedContext]:
        """Generate baseline contexts with preceding words."""
        text = np.array(ds.data)
        all_contexts = []

        for word_index, word in tqdm(
            enumerate(text), desc=f"Generating baseline contexts for story: {story}"
        ):
            all_contexts.append(MaskedContext(context=" ".join(text[max(0, word_index - window_size) : word_index]), mask_indices=[]))
        return all_contexts

    def _generate_entropy_masks(
        self,
        ds,
        window_size: int = 10,
        amount: int = 3,
        mode: str = "peak",
        story: str = "",
    ) -> List[MaskedContext]:
        """Generate contexts with entropy-based masking."""
        self._load_model()

        text = np.array(ds.data)
        all_contexts = []

        for i in tqdm(
            range(len(text)), desc=f"Generating {mode} contexts for story: {story}"
        ):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append("")
                continue

            word_surprisals = self._calculate_surprisals(text, start_idx, context_words)
            mask_indices = self._get_mask_indices(word_surprisals, amount, mode)

            # print(f"Word index {i}, Context: {' '.join(context_words)}, Surprisals: {word_surprisals}, Mask indices: {mask_indices}")
            all_contexts.append(MaskedContext(context=" ".join(context_words), mask_indices=mask_indices.tolist()))

        return all_contexts

    def _calculate_surprisals(
        self, text: np.ndarray, start_idx: int, context_words: List[str]
    ) -> np.ndarray:
        """Calculate surprisal for each word in the context window."""
        word_surprisals = []

        for j in range(len(context_words)):
            # Build prefix from all text up to this point
            prefix = " ".join(text[: start_idx + j])
            target = context_words[j]

            if not target or not target.strip():
                word_surprisals.append(0.0)
                continue

            # Encode prefix and full sequence
            if prefix:
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                full_ids = self.tokenizer.encode(prefix + " " + target, add_special_tokens=False)
            else:
                prefix_ids = []
                full_ids = self.tokenizer.encode(target, add_special_tokens=False)
            
            target_token_ids = full_ids[len(prefix_ids):]
            
            if not target_token_ids:
                word_surprisals.append(0.0)
                continue

            # Calculate surprisal for each token in the word
            word_surprisal = 0.0
            for k, target_id in enumerate(target_token_ids):
                if k == 0:
                    context_ids = prefix_ids
                else:
                    context_ids = full_ids[:len(prefix_ids) + k]
                
                if context_ids:
                    input_tensor = torch.tensor([context_ids]).to(self.model.device)
                else:
                    # Use BOS token for unconditional probability
                    input_tensor = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    prob = probs[target_id].item()
                    surprisal = -np.log2(prob + 1e-10)
                    word_surprisal += surprisal
            
            # Average surprisal across tokens
            avg_surprisal = word_surprisal / len(target_token_ids)
            word_surprisals.append(avg_surprisal)

        return np.array(word_surprisals)

    def _get_mask_indices(
        self, word_surprisals: np.ndarray, amount: int, mode: str
    ) -> np.ndarray:
        """Get indices to mask based on surprisal values."""
        if mode == "peak":
            return word_surprisals.argsort()[-amount:]
        else:
            return word_surprisals.argsort()[:amount]

    def _generate_random_context(
        self, ds, window_size: int = 10, seed: int = 42, story: str = ""
    ) -> List[str]:
        """Generate randomized contexts for null hypothesis testing."""
        random.seed(seed)
        np.random.seed(seed)

        text = np.array(ds.data)
        all_contexts = []

        for i in tqdm(
            range(len(text)),
            desc=f"Generating Random Context (Null Hypothesis) for: {story}",
        ):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append("")
                continue

            random.shuffle(context_words)
            all_contexts.append(" ".join(context_words))

        return all_contexts
