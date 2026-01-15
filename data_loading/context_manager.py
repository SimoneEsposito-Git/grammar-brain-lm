import numpy as np
from typing import Dict, List

from .context_generator import ContextGenerator, MaskedContext


class ContextManager:
    def __init__(self, context_file: str):
        self.context_file = context_file
        self.contexts = self._load_contexts()
        self.generator = ContextGenerator()

    def _load_contexts(self) -> Dict:
        """Load contexts from file with mask metadata."""
        try:
            data = np.load(self.context_file, allow_pickle=True)
            contexts_dict = {}
            
            # Group files by mode (extract mode from filenames like "peak_contexts" and "peak_masks")
            modes = set()
            for filename in data.files:
                if "_contexts" in filename:
                    mode = filename.replace("_contexts", "")
                    modes.add(mode)
                elif "_masks" in filename:
                    mode = filename.replace("_masks", "")
                    modes.add(mode)
            
            for mode in modes:
                contexts_key = f"{mode}_contexts"
                masks_key = f"{mode}_masks"
                
                if contexts_key in data.files:
                    contexts_by_story = data[contexts_key].item()
                    masks_by_story = data[masks_key].item() if masks_key in data.files else {}
                    
                    # Reconstruct MaskedContext objects
                    contexts_dict[mode] = {}
                    for story in contexts_by_story:
                        context_list = contexts_by_story[story]
                        mask_list = masks_by_story.get(story, [[] for _ in context_list]) if masks_by_story else [[] for _ in context_list]
                        
                        contexts_dict[mode][story] = [
                            MaskedContext(ctx, mask) 
                            for ctx, mask in zip(context_list, mask_list)
                        ]
            
            return contexts_dict
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
            self.contexts[mode][story] = self.generator.generate_context(
                dataseqs[story], mode, story=story, **kwargs
            )

        self._save_contexts()

    def _save_contexts(self):
        """Save contexts to disk with mask metadata."""
        save_dict = {}
        for mode, mode_data in self.contexts.items():
            # Extract contexts and mask indices separately
            contexts_by_story = {}
            masks_by_story = {}
            
            for story, context_list in mode_data.items():
                contexts = []
                masks = []
                for masked_ctx in context_list:
                    if isinstance(masked_ctx, MaskedContext):
                        contexts.append(masked_ctx.context)
                        masks.append(masked_ctx.mask_indices)
                    else:
                        # Backward compatibility: handle raw strings
                        contexts.append(masked_ctx)
                        masks.append([])
                
                contexts_by_story[story] = contexts
                masks_by_story[story] = masks
            
            # Save contexts and masks with suffixes
            save_dict[f"{mode}_contexts"] = np.array(contexts_by_story, dtype=object)
            save_dict[f"{mode}_masks"] = np.array(masks_by_story, dtype=object)
        
        np.savez(self.context_file, **save_dict)
