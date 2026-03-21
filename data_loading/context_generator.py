import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List
from english_words import get_english_words_set
import spacy

POS_TAGS = ["noun", "verb", "adj", "adv", "pron", "aux", "propn","intj"]  # Common POS tags to consider for masking
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

    def generate_mask(
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
    
    def generate_context(
        self,
        ds,
        mode: str,
        window_size: int = 10,
        amount: int = 3,
        seed: int = 42,
        story: str = "",
        **kwargs,
    ) -> List[str]:  
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
            return self._generate_surprise_masks(ds, word_surprisals[story], window_size, amount, mode, story)
        elif "remove-pos-pct" in mode:
            pos_tags = mode.split("-")[3:]  # Extract POS tags from mode string like "remove-pos-pct-VERB"
            return self._generate_pos_pct_masks(ds, pos_tags, window_size, story)
        elif "remove-pos" in mode:
            pos_tags = mode.split("-")[2:]  # Extract POS tags from mode string
            if 'verb' in pos_tags:
                pos_tags.append('aux')  # Include auxiliary verbs as well
            if 'noun' in pos_tags:
                pos_tags.append('propn')  # Include proper nouns as well
            return self._generate_pos_masks(ds, pos_tags, window_size, story)
        elif mode == "zero":
            return ["" for _ in range(len(ds.data))]  
        elif mode == "random":
            return self._generate_random_context(ds, window_size, seed, story)
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
            context = " ".join(text[max(0, word_index - window_size) : word_index])
            all_contexts.append(context)  # Append the context string directly
        return all_contexts

    def _generate_surprise_masks(
        self,
        ds,
        word_surprisals: np.ndarray,
        window_size: int = 10,
        amount: int = 3,
        mode: str = "remove-peaks",
        story: str = "",
    ) -> List[str]:
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
        
    def _generate_pos_masks(
        self,
        ds,
        pos_tags: List[str],
        window_size: int = 10,
        story: str = "",
    ) -> List[str]:
        """Generate contexts with POS-based masking."""
        nlp = spacy.load("en_core_web_sm")
        text = np.array(ds.data)
        
        # First pass: identify POS tags for the entire text
        full_text = " ".join(text.tolist())
        doc = nlp(full_text)
        
        # Map original word indices to their POS tags by matching tokens to words
        word_pos_tags = []
        word_idx = 0
        for token in doc:
            if word_idx < len(text):
                word_pos_tags.append(token.pos_.lower())
                word_idx += 1
        
        all_contexts = []
        stop_idx = 0
        for i in tqdm(
            range(len(text)),
            desc=f"Generating POS-based masks for story: {story}",
        ):
            start_idx = max(0, stop_idx - window_size)
            context_tokens = doc[start_idx:stop_idx]
            context_words = []
            stop_idx += nlp(str(text[i])).__len__() # Update stop_idx based on token count of current word
            
            # Mask words in context window that match the specified POS tags
            for token in context_tokens:
                if token.pos_.lower() in pos_tags:
                    token_idx = token.i - start_idx
                    context_words.append("XXXX")
                else:
                    context_words.append(token.text)
            all_contexts.append(" ".join(context_words))
        return all_contexts
    
    def _generate_pos_pct_masks(
        self,
        ds,
        pos_tags: List[str],
        window_size: int = 10,
        story: str = "",
    ) -> List[str]:
        """Generate contexts with random masking based on POS tag count in each context window.
        
        For each word, calculates how many words in its context window have the specified POS tags,
        then masks that same number of random words instead.
        """
        nlp = spacy.load("en_core_web_sm")
        text = np.array(ds.data)
        all_contexts = []

        for i in tqdm(
            range(len(text)),
            desc=f"Generating POS-percentage-based masks for story: {story}",
        ):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append("")
                continue

            # Parse context to get POS tags
            context_text = " ".join(context_words)
            context_doc = nlp(context_text)

            # Count how many words match the specified POS tags
            matching_count = sum(1 for token in context_doc if token.pos_.lower() in pos_tags)

            # Randomly select the same amount of indices to mask
            if matching_count > 0:
                mask_limit = min(matching_count, len(context_words))
                mask_indices = random.sample(range(len(context_words)), mask_limit)
                for idx in mask_indices:
                    context_words[idx] = "XXXX"

            all_contexts.append(" ".join(context_words))

        return all_contexts
    
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
    ) -> List[str]:
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
