import numpy as np
import spacy
from typing import Dict, List

POS_TAGS = ["noun", "verb", "adj", "adv", "pron", "aux", "propn","intj"]  # Common POS tags to consider for masking

class ContextManager:
    def __init__(self, context_file: str):
        self.context_file = context_file
        self.contexts = self._load_contexts()

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

    def _generate_and_cache(
        self, stories: List, mode: str, dataseqs: Dict, **kwargs
    ) -> None:
        """Generate contexts for missing stories and save to disk."""
        for story in stories:
            self.contexts[mode][story] = self._generate_mask(
                dataseqs[story], mode, story=story, **kwargs
            )
        self._save_contexts()
    
    def _generate_mask(
        self,
        ds,
        mode: str,
        seed: int = 42,
        story: str = "",
        **kwargs,
    ) -> List[bool]:  
        """Generate contexts based on the specified mode."""
        if mode == "baseline":
            return [False] * len(ds.data)  # No masking, keep all words
        elif mode in POS_TAGS:
            pos_tags = [mode]
            if 'verb' in pos_tags:
                pos_tags.append('aux')  # Include auxiliary verbs as well
            if 'noun' in pos_tags:
                pos_tags.append('propn')  # Include proper nouns as well
            return self._generate_pos_masks_bool(ds, pos_tags, story)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _generate_pos_masks_bool(
        self,
        ds,
        pos_tags: List[str],
        story: str = "",
    ) -> List[bool | List[bool]]:
        """Generate boolean mask for words matching specified POS tags."""
        nlp = spacy.load("en_core_web_sm")
        text = [t.strip() for t in ds.data]
        full_text = " ".join(text)
        doc = nlp(full_text)

        # Group spaCy tokens back into original whitespace-separated words
        words = full_text.split(" ")
        mask = []
        token_iter = iter(doc)

        for word in words:
            # Consume spaCy tokens until we've reconstructed the original word
            accumulated = ""
            token_masks = []
            
            while accumulated != word:
                try:
                    token = next(token_iter)
                    accumulated += token.text
                    token_masks.append(token.pos_.lower() in pos_tags)
                except StopIteration:
                    break

            # If spaCy split the word into multiple tokens (e.g. contraction), return a list
            if len(token_masks) == 1:
                mask.append(token_masks[0])
            else:
                mask.append(token_masks)
        assert len(mask) == len(text), "Mask mismatch: expected length {}, got {}".format(len(text), len(mask))
        return mask
        
    def _save_contexts(self):
        """Save contexts to disk."""
        np.savez(
            self.context_file,
            **{k: np.array(v, dtype=object) for k, v in self.contexts.items()},
        )
