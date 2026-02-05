# 2. Background

## Voxelwise encoding modelling

This thesis uses a method known as voxelwise encoding models, in which BOLD responses of each individual voxel are modelled as a linear combinations of specific features. When the prediction accuracy for each voxel is then mapped onto the brain, areas that are better predicted may indicate that the corresponting part of the brain might encode information of the given features.  The features themselves are generally nonlinear transformations of the stimuli onto a high dimensional feature space. 

One prominent model (Huth2016) constructs the features by computing for each word in the stimulus the co-occurence between that word and 985 basic english words based on a large corpus of texts. This embeds each word in the stimulus onto a 985 dimensional vector, where similar words tend to correlate. This model has proven to be highly effective at predicting brain activity. 

## Contextual embeddings
Nevertheless, the main shortcoming of that model is that the same word will always be mapped to the same embedding, regardless of context. For example the word *fine* in *"I am fine", "I have to pay a fine"* and *"I used a fine brush"* all have completely different meanings which might dilute the information in the features. 

To address this, language models (LM) started to be used to generate contextual embeddings. First people started by using Long Short-Term Memory networks (LSTM) [Jain, et al 2018], then as technology improved, more advanced language models such as GPT-2 or Llama started to be used to generate such embeddings. 

To show that these contextual models actually relied on the context, the latter was scrambled and randomized which as expected lead to an overall worse performance. Nevertheless there has not been a paper taking a closer look on what effect other kinds of context manipulation might have on prediction accuracy



# 3. Materials

## 3.1 Stimuli 

The speech stimuli consist of 11 short stories taken from **The Moth Radio Hour**, a storytelling program where true autobiographical stories are told in front of a live audience. These stimuli were used in previous papers (huth2016), including the one where the response data was gathered (deniz2019). These stories provide a naturalistic stimulus.

Out of these 11 stories, 10 were used as training data while one 10 minute story was used as validation data. The latter was played two times during the experiment (in two separate scans) to which the responses were finally averaged across the two runs.

These stories were then transcribed and stored as Praat's TextGrid object (Boersma and
Weenink, 2001), a text-based file format containing the transcription of the words spoken in the stories and the times where they were spoken. Additionally, the text grids also contained phoneme information which was not used in the project. 

## 3.2 fMRI responses

Each spoken (and written) story was presented during a separate fMRI scan. The length of each scan was the same as the story. The sampling rate was 2 seconds, so $1 \text{ TR} = 2 s$. Each scan included 10 seconds (5 TR) of silence both before and after the story. These data were collected during 23h scanning sessions that were performed on different days. MRI data were collected on a 3T Siemens TIM Trio scanner at the UC Berkeley Brain Imaging Center using a 32-channel Siemens volume coil. 

Each functional run was motion-corrected using the FMRIB Linear Image Registration Tool (FLIRT) [expand?] from FSL 5.0 (Jenkinson and Smith, 2001; Jenkinson et al., 2002). All volumes in the run were then averaged across time to obtain a high quality template volume. FLIRT was also used to automatically align the template volume for each run to the overall template, which was chosen to be the temporal average of the first functional run for each participant. 

The responses were provided as HDF files. For each subject and each modality there were two files: one for the training stories and one for the validation story, for a total of 36 files. Each training file contains a dictionary with the stories as keys and the BOLD responses as values. Each validation file contains a dictionary with *story_11* as the key and the responses from both repetitions as values.

The responses are represented as an 2d `numpy` array with shape $(N_{TR} \times N_{voxels})$. The amount of TRs depends on the story while the amount of voxels is unique to the subject. The response data also included the 10 seconds of silence before and after the story was played.

## 3.3 Features

To reproduce the original study, the 985–dimensional Eng1000 features were provided

To account for responce variance caused by various modality–dependent nuisances, several additional features were provided.

### Phonoemes 
A list of phonome-time pairs was constructed for each of the 39 english phonemes. This list was then down–sampled to the fMRI aquisition rate. The result is a $(39\times N_{TR})$ array. These features were used when fitting the model to the *listening* responses

### Letters
Similar to the phonemes, a 26 parameter model was created to represent the frequency of each of the 26 letters in the english alphabet that appeared in the written stories. This was also down–sampled to the sampling rate, resulting in a $(26\times N_{TR})$ array.

### Additional Features
To account for the highly variable speech rate both within and across
stories, single-feature models that simply count the number of words,
number of phonemes, number of letters, and number of story speaker’s
pauses that occurred during the acquisition of each fMRI volume (2.0045
s) were constructed. To account for the variable word lengths during the
visual presentation a single-feature word length variation model was
constructed by taking the variance of word lengths that occurred during
the acquisition of each fMRI volume

## 3.4 Brain Mappers

Anatomical and functional data were not provided due to privacy regulations. Instead, pre-computed mapper files (HDF) were used, containing for each subject the data to map the voxels onto flattened representations of the brain, facilitating the generation of cortical flatmaps via the `matplotlib` library. 

The file contains four components that are used to load a *compressed sparse row matrix*, a type used to store matrices containing mostly zero or NaN values in an efficient way by only storing the relevant values and their coordinates in the matrix. Specifically:

1. `voxel_to_flatmap_data` contains the non-zero data. In our case it's a list of ones. 

2.  `voxel_to_flatmap_indices` contains the column index for each value in the data array. It has the same size as the data array.

3.  `voxel_to_flatmap_indptr` contains the row pointers. More precisely, $\text{indptr}_i$ represents the starting position in the data array where row $i$ begins while $\text{indptr}_{i+1}$ represents the ending position (exclusive) where row $i$ ends. The number of non-zero elements in row i is given by: $\text{indptr}_{i+1}-\text{indptr}_{i}$ 

4.  `voxel_to_flatmap_shape` contains the shape of the sparse matrix
   
This results in a sparse matrix $M$ with each row representing one pixel of the flattened brain image *excluding* the background pixels and each column representing one voxel of the brain. 

This means, $M_{i,j}=1$ if voxel $j$ is mapped to pixel $i$. these indices represent the flattened coordinates of the brain image's pixels. 

The use of the Compressed Sparse Row (CSR) format is computationally motivated by the high sparsity of the mapping matrix M. In a typical fMRI volume, only a fraction of voxels corresponds to the cortical surface, and each voxel is mapped to only one or a few pixels in the flattened 2D representation

Finally, an additional component `flatmap_mask` contains a matrix  $I$ with the rows and columns representing the width and height of the flatmap image. In this mask matrix, $I_{x,y} = 1$ if the pixel at coordinate $(x,y)$ is part of the brain. Consequently, the values of this mask add up to the size of the first axis of $M$. 

The sparse matrix and the mask can then be used to map the prediction scores of each voxel onto the final flatmap index. The details are explained in the [visualization](#49-visualization) section.

# 4. Methods 
## 4.1 Loading Features and Responses 
In the first step of the pipeline, the features and responses, both training and validation, were extracted from the respective HDF files using the python library `h5py`. The features and responses were stored in a python dictionary with the following structures:

```python
F_trn = {
    'story_01':{
        'english1000': array(shape = (n_tr_story_01, 985)),
        'letters' : array(shape = (n_tr_story_01, 26)),
        ...
    }
    'story_02':{
        ...
    }
}
# F_val analogous to F_trn but containing only 

R_trn = {
    'subject_01':{
        'story_01': array(
            shape = (n_tr_story_01, n_voxels_subject_01)
        ),
        ...
    }
    'subject_02':{
        ...
    }
}

R_val = {
    'subject_01':{
        'story_11': array(
            shape = (2, n_tr_story_01, n_voxels_subject_01)
        )
    },
    ...
}
```

During this process, nuisance features of both modalities were stored. When modelling using the Eng1000 features, the preprocessing process would end here.

## 4.2 Data Sequence Generation and Loading
Since embeddings need to be generated on a continuous word-by-word basis, but the features need to be considered at the level of TRs and thus in discrete chunks, a specialized class called DataSequence was used to store stimulus data, and later, embedding data. 

A Data Sequence included the following attributes and methods:

- `data`: an array of all data points, e.g.: the words in the story
- `split_inds`: the indices that divided different chunks of `data`
- `data_times`: the time in seconds of each data point 
- `tr_times`: the time in seconds of each TR
- `chunksums: (interp, **kwargs)->array`: a method that returns the `data`, splits it into the discrete chunks (at `split_inds`), takes the sum of each chunks and returns the new matrix with the rows reduced to one per chunk. If interp is provided, the method will downsample the data with the respective filter. The options available are: *Sinc*, *Gabor* and *Lanczos*, the latter being the one used for the contextual embeddings generation

This class also included various *classmethods*, i.e. methods bound to the class instead of the instance of the class, that in turn themselves returned an instance of the class. The method `DataSequence.from_grid(cls, grid_transcript, trfile, word_time)` was used to generate Data Sequeces using the given *Textgrids* and *TR Files*, the latter being a file storing all the time points. 

Since the loading of the original stimuli into the DataSequence objects could be lengthy, and only needed to be done once, a one-time script was employed to generate the DataSequence objects and store them as `pickle` files (.pkl), which could then be directly loaded in the pipeline.

## 4.3 Context Generation
The core of this thesis is to test the effect of intentional disruption of the contextual LLM embeddings by manipulating the context itself. 

The contexts are provided in the form of a dictionary, with the key being the story and the value being a list of the manipulated contexts leading to the target word, excluding the target word itself. 

To streamline the process, I created a specialized `ContextManager` class, to generate new contexts or load existing ones. The singular public method `get_or_create_contexts` takes in the following parameters:

- `mode: str`: the identifier for the type of manipulation used.
- `stories: List`: the stories used.
- `dataseqs: Dict`: the unmodified stimulus data sequences, with the stories as dictionary keys.
- `overwrite: bool`: whether or not to override existing contexts. Used primarily in the developement stage.
- `**kwargs`: additional mode-specific arguments

And returns the contexts in the aforementioned structure.

If there is no existing context and overwrite is flagged `False`, the method calls the `generate_context(dataseqs[story], mode, story, **kwargs)` for each story. This method in turn uses if-else statemens on the modes to call private mode-specific methods that return the contexts for the given story. 

### 4.3.1 Baseline
This method is used to generate the unmodified contexts used for the generation of the conventional contextual embeddings. These embeddings were used for comparison purpuse.
```python
def _generate_baseline_context(
        self, ds, 
        window_size: int = 10
    ) -> List[str]:

    text = np.array(ds.data)
    all_contexts = []
    for word_index in range(len(text)):
        context = " ".join(
            text[
                max(0,word_index -window_size) 
                :word_index
            ]
        )
        all_contexts.append(context)  
    return all_contexts
```

### 4.3.2 Random
This method is used to generate a random context, consisting of $n$ random words from the english dictionary using the `english_words` library. This was used to show that the model does in fact rely on contextual information, as done in the original paper for contextual embeddings [Jain, et al 2018].

```python
def _generate_random_context(
    self, ds, 
    context_size: int = 10, 
    seed: int = 42
) -> List[str]:
    random.seed(seed)
    text = np.array(ds.data)
    dict_set = get_english_words_set(['gcide'], lower=True)
    all_contexts = []
    
    for _ in range(text):
        all_contexts.append(" ".join(
            random.sample(list(dict_set), context_size)
        ))
    return all_contexts
```

### 4.3.3 Shuffle
This method is used to generate contexts similar to the baseline contexts, but with the order of the words randomly shuffled. This was used to see whether the order of the words in the context hurt the model. This was also done in the original paper.

```python
def _generate_shuffle_context(
    self, ds, 
    window_size: int = 10,
    seed: int = 42
) -> List[str]:
    random.seed(seed)
    text = np.array(ds.data)
    all_contexts = []
    for i in range(len(text)):
        start_idx = max(0, i - window_size)
        context_words = text[start_idx:i].tolist()
        if not context_words:
            all_contexts.append("")
            continue
        random.shuffle(context_words)
        all_contexts.append(" ".join(context_words))
    return all_contexts
```

### 4.3.4 Surprisal Masks
The first two real manipulations I tried involved masking the most and the least surprising words in the context. These modes were called *remove_peaks* and *remove_valleys*.

To generate these embeddings I first needed to calculate the 'surprisal' of each word.

Fortunately, the same Language Models used to generate the contextual embeddings are also used to do Next Word Prediction. Given a sequence of tokens $t_1\dots t_i$ the model will generate the next token $t_{i+1}$ by sampling from the probability distribution $P(t_{i+1}\ |\  t_1, \dots t_i)$. The hidden state $h_i$ at position $i$ is linearly projected into a vector of vocabulary-sized logits:
$$z = h_iW+b$$
 Applying *softmax* yields the conditional probability distribution:
$$P(t_{i+1}\ |\  t_1, \dots t_i) = \mathrm{softmax}(z)$$

The surprisal (Self-information) of a token $t$ is then
$$S(t) = -\log_2 p_t$$

When it comes to removing the *most* or *least* surprising words, one could use the logits directly, since both the softmax and log function are monotonic, and thus the order is preserved. I still decided to use the surprisal values since this would allow more options for later experiments like masking words above a certain information threshold. 

To extract the logits from each word in the text and calculate the surprisal I used the following code.

```python
def _calculate_surprisals(self, text):
    word_surprisal = []

    for j in range(len(text)):
        # assigning a surprisal of 0 to avoid a bias towards the first word, 
        # since the surprisal tends to be overly high
        if j == 0:
            word_surprisals.append(0.0)
            continue

        # Build prefix from all text up to this point
        prefix = " ".join(text[:j])
        target = text[j]

        # Encode prefix and full sequence
        if prefix:
            prefix_ids = self.tokenizer.encode(prefix)
            full_ids = self.tokenizer.encode(prefix + " " + target)
        else:
            prefix_ids = []
            full_ids = self.tokenizer.encode(target)

        # Extract the target token ids
        target_token_ids = full_ids[len(prefix_ids):]

        # Calculate surprisal for each token in the word
        word_surprisal = 0.0
        for k, target_id in enumerate(target_token_ids):
            context_ids = full_ids[:len(prefix_ids) + k]
            # Transfer ids to gpu 
            input_tensor = torch.tensor([context_ids])
                                .to(self.model.device)
        
            # Calculate surprisal as described above
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                prob = probs[target_id].item()
                surprisal = -np.log2(prob)
                word_surprisal += surprisal
        
        # Average surprisal across tokens
        avg_surprisal = word_surprisal / len(target_token_ids)
        word_surprisals.append(avg_surprisal)

    return word_surprisals
```
This method was then used to mask the original context by using the `word_surprisal[s:t].argsort()` function – with `s` and `t` spanning the context window size – to get the indices of the $n$ most `[-n:]` and the $n$ least `[:n]` surprising words in the context. The context words at those indices were then replaced with *XXXX*
```python 
def _generate_entropy_masks(
        self,
        ds,
        word_surprisals: np.ndarray,
        window_size: int = 10,
        n: int = 3,
        mode: str = "remove-peaks",
    ) -> List[str]:
        text = np.array(ds.data)
        all_contexts = []
            
        for i in range(len(text)):
            start_idx = max(0, i - window_size)
            context_words = text[start_idx:i].tolist()

            if not context_words:
                all_contexts.append('') 
                continue
            
            # Ensure first word is not masked
            if mode == "remove-peaks" or 
               mode == "valleys-only":
                word_surprisals[0] = 0  

            if mode == "remove-valleys" or 
               mode == "peaks-only":
                word_surprisals[0] = float('inf')  
            
            # At least one word in the context should be unmasked
            n_ = min(n, len(context_words)-1)

            mask = np.array([], dtype=int)

            if mode == "remove-peaks":
                mask =  word_surprisals.argsort()[-n_:]
            elif mode == "remove-valleys":
                mask =  word_surprisals.argsort()[:n_]

            # Replace masked words with XXXX
            for idx in mask_indices:
                context_words[idx] = "XXXX"
            all_contexts.append(" ".join(context_words))  # Append the modified context

        return all_contexts
```



## 4.3 Embedding Generation
## 4.4 Feature Preparation
## 4.5 Response Preparation
## 4.6 Multiple Kernel Ridge Regression
## 4.7 Permutation Testin
## 4.8 FDR Correction
## 4.9 Visualization

# 5. Results

# 6. Discussion