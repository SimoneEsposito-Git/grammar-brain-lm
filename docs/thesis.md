# Background

## Voxelwise encoding modelling

This thesis uses a method known as voxelwise encoding models, in which BOLD responses of each individual voxel are modelled as a linear combinations of specific features. When the prediction accuracy for each voxel is then mapped onto the brain, areas that are better predicted may indicate that the corresponting part of the brain might encode information of the given features.  The features themselves are generally nonlinear transformations of the stimuli onto a high dimensional feature space. 

One prominent model (Huth2016) constructs the features by computing for each word in the stimulus the co-occurence between that word and 985 basic english words based on a large corpus of texts. This embeds each word in the stimulus onto a 985 dimensional vector, where similar words tend to correlate. This model has proven to be highly effective at predicting brain activity. 

## Contextual embeddings
Nevertheless, the main shortcoming of that model is that the same word will always be mapped to the same embedding, regardless of context. For example the word *fine* in *"I am fine", "I have to pay a fine"* and *"I used a fine brush"* all have completely different meanings which might dilute the information in the features. 

To address this, language models (LM) started to be used to generate contextual embeddings. First people started by using Long Short-Term Memory networks (LSTM) [Jain, et al 2018], then as technology improved, more advanced language models such as GPT-2 or Llama started to be used to generate such embeddings. 

To show that these contextual models actually relied on the context, the latter was scrambled and randomized which as expected lead to an overall worse performance. Nevertheless there has not been a paper taking a closer look on what effect other kinds of context manipulation might have on prediction accuracy



# Materials and Methods    
## Stimuli 

The speech stimuli consist of 11 short stories taken from **The Moth Radio Hour**, a storytelling program where true autobiographical stories are told in front of a live audience. These stimuli were used in previous papers (huth2016), including the one where the response data was gathered (deniz2019).  
Out of these 11 stories, 10 were used as training data while one 10 minute story was used as validation data. The latter was played two times during the experiment (in two separate scans) to which the responses were finally averaged across the two runs.

These stories were then transcribed and stored as Praat's TextGrid object (Boersma and
Weenink, 2001), a text-based file format containing the transcription of the words spoken in the stories and the times where they were spoken. Aditionally, the text grids also contained phomene information which was not used in the project. 

# fMRI responses

 

