import numpy as np
from typing import Dict, List

from .context_generator import ContextGenerator


class ContextManager:
    def __init__(self, context_file: str):
        self.context_file = context_file
        self.contexts = self._load_contexts()
        self.generator = ContextGenerator()

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
            self.contexts[mode][story] = self.generator.generate_mask(
                dataseqs[story], mode, story=story, **kwargs
            )
        self._save_contexts()
        
    def _generate_and_cache_(self, stories: List, mode: str, dataseqs: Dict, **kwargs):
        """Generate contexts for missing stories and save to disk."""
        for story in stories:
            self.contexts[mode][story] = self.generator.generate_context(
                dataseqs[story], mode, story=story, **kwargs
            )

        self._save_contexts()

    def _save_contexts(self):
        """Save contexts to disk."""
        np.savez(
            self.context_file,
            **{k: np.array(v, dtype=object) for k, v in self.contexts.items()},
        )
