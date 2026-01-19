import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List
from english_words import get_english_words_set

class ContextGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_seq_length = 1023
        
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
        **kwargs,
    ) -> List[str]:  # Change return type to List[str]
        """Generate contexts based on the specified mode."""
        if mode == "shuffle":
            return self._generate_shuffle_context(ds, window_size, seed, story)
        elif mode == "baseline":
            return self._generate_baseline_context(ds, window_size, story)
        elif mode in ["remove-peaks", "remove-valleys", "peaks-only", "valleys-only"]:
            surprisals_file = kwargs.get('surprisals_file', None)
            try:
                word_surprisals = np.load(surprisals_file, allow_pickle=True).item()
            except FileNotFoundError:
                word_surprisals = {}
                
            if story not in word_surprisals:
                word_surprisals[story] = self._calculate_surprisals(ds.data, story)
            np.save(surprisals_file, word_surprisals)
            return self._generate_entropy_masks(ds, word_surprisals[story], window_size, amount, mode, story)
        elif mode == "zero":
            return ["XXXX" for _ in range(len(ds.data))]  
        elif mode == "random":
            return self._generate_random_context(ds, window_size, seed, story)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_baseline_context(
        self, ds, window_size: int = 10, story: str = ""
    ) -> List[str]:  # Change return type to List[str]
        """Generate baseline contexts with preceding words."""
        text = np.array(ds.data)
        all_contexts = []

        for word_index, word in tqdm(
            enumerate(text), desc=f"Generating baseline contexts for story: {story}"
        ):
            context = " ".join(text[max(0, word_index - window_size) : word_index])
            all_contexts.append(context)  # Append the context string directly
        return all_contexts

    def _generate_entropy_masks(
        self,
        ds,
        word_surprisals: np.ndarray,
        window_size: int = 10,
        amount: int = 3,
        mode: str = "remove-peaks",
        story: str = "",
    ) -> List[str]:  # Change return type to List[str]
        """Generate contexts with entropy-based masking."""
        self._load_model()

        text = np.array(ds.data)
        all_contexts = []
            
        for i in range(len(text)):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append('') 
                continue
            
            if mode == "remove-peaks" or mode == "valleys-only":
                word_surprisals[0] = 0  # Ensure first word is not masked
            if mode == "remove-valleys" or mode == "peaks-only":
                word_surprisals[0] = float('inf')  # Ensure first word is not masked
            
            amount_ = min(amount, len(context_words)-1)
            mask_indices = self._get_mask_indices(word_surprisals[start_idx:i], amount_, mode)

            # Replace masked words with <MASK>
            for idx in mask_indices:
                context_words[idx] = "XXXX"
            all_contexts.append(" ".join(context_words))  # Append the modified context

        return all_contexts

    def _calculate_surprisals(
        self, text: np.ndarray, story: str = ""
    ) -> np.ndarray:
        """Calculate surprisal for each word in the context window."""
        word_surprisals = []

        for j in tqdm(range(len(text)), desc=f"Calculating word surprisals for story: {story}"):
            if j == 0:
                word_surprisals.append(0.0)
                continue
            # Build prefix from all text up to this point
            prefix = " ".join(text[:j])
            target = text[j]

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
                
                # Truncate to max sequence length
                if context_ids:
                    context_ids = context_ids[-self.max_seq_length + 1:]
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
        if amount == 0:
            return np.array([], dtype=int)
        if mode == "remove-peaks":
            return word_surprisals.argsort()[-amount:]
        elif mode == "remove-valleys":
            return word_surprisals.argsort()[:amount]
        elif mode == "peaks-only":
            return word_surprisals.argsort()[:-amount]
        elif mode == "valleys-only":
            return word_surprisals.argsort()[amount:]

    def _generate_shuffle_context(
        self, ds, window_size: int = 10, seed: int = 42, story: str = ""
    ) -> List[str]:
        """Generate randomized contexts for null hypothesis testing."""
        random.seed(seed)
        np.random.seed(seed)

        text = np.array(ds.data)
        all_contexts = []

        for i in tqdm(
            range(len(text)),
            desc=f"Generating Random Context for: {story}",
        ):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append("")
                continue

            random.shuffle(context_words)
            all_contexts.append(" ".join(context_words))

        return all_contexts

    def _generate_random_context(
        self, ds, context_size: int = 10, seed: int = 42, story: str = ""
    ):
        random.seed(seed)
        np.random.seed(seed)
        dict_set = get_english_words_set(['gcide'], lower=True)

        text = np.array(ds.data)
        all_contexts = []
        
        for i in tqdm(
            range(len(text)),
            desc=f"Generating Random Context for: {story}",
        ):
            all_contexts.append(" ".join(random.sample(list(dict_set), context_size)))
        return all_contexts