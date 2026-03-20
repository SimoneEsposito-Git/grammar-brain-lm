from typing import Dict, List, Optional
import numpy as np


class EmbeddingManager:
    def __init__(self, embeddings_file: str):
        self.embeddings_file = embeddings_file
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self) -> Dict:
        """Load embeddings from file or create new dict."""
        try:
            data = np.load(self.embeddings_file, allow_pickle=True)
            return {k: data[k].item() for k in data.files}
        except (FileNotFoundError, EOFError) as e:
            print(f"Warning: Creating new embeddings. {e}")
            return {}

    def get_or_create_embeddings(
        self,
        mode: str,
        stories: List,
        dataseqs: Dict,
        contexts: Dict,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """Get cached embeddings or generate new ones for missing stories."""
        
        if mode not in self.embeddings or self.embeddings[mode] is None or overwrite:
            self.embeddings[mode] = {}

        missing_stories = [s for s in stories if s not in self.embeddings[mode]]
        print(f"Mode '{mode}': {len(missing_stories)} missing stories out of {len(stories)}.")
        if missing_stories:
            self._generate_and_cache(
                missing_stories,
                mode,
                dataseqs,
                contexts,
                verbose=verbose,
            )

        return self.embeddings[mode]

    def _generate_and_cache_(self, stories, mode, dataseqs, contexts, verbose):
        """Generate embeddings for missing stories and save to disk."""
        from .embedding_generator import contextual_embeddings

        new_emb = contextual_embeddings(
            dataseqs,
            "openai-community/gpt2",
            8,
            interp="lanczos",
            contexts=contexts,
            verbose=verbose,
        )
        for story in stories:
            self.embeddings[mode][story] = new_emb[story]

        self._save_embeddings()
        
    def _generate_and_cache(self, stories, mode, dataseqs, mask, verbose=False, window_size=20):
        """Generate embeddings for missing stories and save to disk."""
        from .embedding_generator import masked_embeddings
        new_emb = masked_embeddings(
            dataseqs,
            "openai-community/gpt2",
            8,  
            interp="lanczos",
            mask=mask,
            window_size=window_size,
            verbose=verbose,
        )
        for story in stories:
            print(f"Caching embeddings for story {story} in mode {mode}.")
            self.embeddings[mode][story] = new_emb[story]

        self._save_embeddings()
    def _save_embeddings(self):
        """Save embeddings to disk."""
        np.savez(
            self.embeddings_file,
            **{k: np.array(v, dtype=object) for k, v in self.embeddings.items()},
        )
