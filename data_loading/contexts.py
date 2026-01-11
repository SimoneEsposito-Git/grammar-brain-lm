import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List


class ContextManager:
    def __init__(self, context_file: str):
        self.context_file = context_file
        self.contexts = self._load_contexts()
        self.tokenizer = None
        self.model = None

    def _load_contexts(self) -> Dict:
        """Load contexts from file or create new dict."""
        try:
            data = np.load(self.context_file, allow_pickle=True)
            return {k: data[k].item() for k in data.files}
        except (FileNotFoundError, EOFError) as e:
            print(f"Warning: Creating new contexts. {e}")
            return {}

    def get_or_create_contexts(
        self,
        mode: str,
        stories: List,
        dataseqs: Dict,
        overwrite: bool = False,
        **kwargs,
    ) -> Dict:
        """Get cached contexts or generate new ones for missing stories.

        Args:
            mode: Context generation mode ("random", "baseline", "peak", "valley", "zero").
            stories: List of story identifiers.
            dataseqs: Dictionary mapping story identifiers to DataSequence objects.
            overwrite: If True, regenerate all contexts for this mode.
            **kwargs: Additional arguments passed to context generation (window_size, amount, seed, etc).

        Returns:
            Dictionary mapping story identifiers to their context lists.
        """
        if mode not in self.contexts or self.contexts[mode] is None or overwrite:
            self.contexts[mode] = {}

        missing_stories = [s for s in stories if s not in self.contexts[mode]]

        if missing_stories:
            self._generate_and_cache(missing_stories, mode, dataseqs, **kwargs)

        return self.contexts[mode]

    def _generate_and_cache(self, stories: List, mode: str, dataseqs: Dict, **kwargs):
        """Generate contexts for missing stories and save to disk."""
        for story in stories:
            self.contexts[mode][story] = self._generate_context(
                dataseqs[story], mode, story=story, **kwargs
            )

        self._save_contexts()

    def _save_contexts(self):
        """Save contexts to disk."""
        np.savez(
            self.context_file,
            **{k: np.array(v, dtype=object) for k, v in self.contexts.items()},
        )

    def _load_model(self):
        """Lazy load GPT-2 model and tokenizer."""
        if self.model is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")

    def _generate_context(
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
        elif mode == "zero":
            return [""] * len(ds.data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_baseline_context(
        self, ds, window_size: int = 10, story: str = ""
    ) -> List[str]:
        """Generate baseline contexts with preceding words."""
        text = np.array(ds.data)
        all_contexts = []

        for word_index, word in tqdm(
            enumerate(text), desc=f"Generating baseline contexts for story: {story}"
        ):
            all_contexts.append(
                " ".join(text[max(0, word_index - window_size) : word_index + 1])
            )

        return all_contexts

    def _generate_entropy_masks(
        self,
        ds,
        window_size: int = 10,
        amount: int = 3,
        mode: str = "peak",
        story: str = "",
    ) -> List[str]:
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

            masked_words = context_words.copy()
            for idx in mask_indices:
                masked_words[idx] = "[MASK]"

            all_contexts.append(" ".join(masked_words))

        return all_contexts

    def _calculate_surprisals(
        self, text: np.ndarray, start_idx: int, context_words: List[str]
    ) -> np.ndarray:
        """Calculate surprisal for each word in the context window."""
        word_surprisals = []

        for j in range(len(context_words)):
            prefix = " ".join(text[start_idx : start_idx + j])
            target = context_words[j]

            if not target or not target.strip():
                word_surprisals.append(0.0)
                continue

            target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
            if not target_tokens:
                word_surprisals.append(0.0)
                continue

            inputs = self.tokenizer(
                prefix if prefix else target, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

            target_id = target_tokens[0]
            p_word = probs[target_id].item()
            surprisal = -np.log2(p_word + 1e-10)
            word_surprisals.append(surprisal)

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


# Backwards compatibility wrapper
def generate_context(ds, mode, **kwargs):
    """Legacy function for backwards compatibility."""
    manager = ContextManager(context_file="contexts.npz")
    return manager._generate_context(ds, mode, **kwargs)
