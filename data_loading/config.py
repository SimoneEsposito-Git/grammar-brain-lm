from pathlib import Path

# Experiment configurations
SUBJECTS = [
    "subject01",
    "subject02",
    "subject03",
    "subject04",
    "subject05",
    "subject06",
    "subject07",
    "subject08",
    "subject09",
]

STORIES_OLD = [
    "story_01",
    "story_02",
    "story_03",
    "story_04",
    "story_05",
    "story_06",
    "story_07",
    "story_08",
    "story_09",
    "story_10",
    "story_11",
]

STORIES = [
    "alternateithicatom_en",
    "avatar_en",
    "howtodraw_en",
    "legacy_en",
    "life_en",
    "myfirstdaywiththeyankees_en",
    "naked_en",
    "odetostepfather_en",
    "souls_en",
    "undertheinfluence_en",
    "wheretheressmoke_en",
]

BAD_WORDS = [
    "{BR}",
    "{LG}",
    "{LS}",
    "{NS}",
    "{CG}",
    "",
    "sp",
    "sentence_start",
    "sentence_end",
]

NUIS_READING = ["letters", "numletters", "word_length_std", "numwords", "pauses"]
NUIS_LISTENING = ["phonemes", "numphonemes", "numwords", "pauses"]

# Default paths
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATASEQ_PATH = DATA_DIR / "stimuli" / "data_sequences"
TEXTGRID_PATH = DATA_DIR / "stimuli" / "textgrids"
TRFILE_PATH = DATA_DIR / "stimuli" / "trfiles"
FEATURE_PATH = DATA_DIR / "features"
EMBEDDINGS_FILE = FEATURE_PATH / "embeddings.npz"
CONTEXTS_FILE = FEATURE_PATH / "contexts.npz"
RESPONSE_PATH = DATA_DIR / "responses"
MAPPER_PATH = DATA_DIR / "mappers"

# Model configurations
MODEL = "openai-community/gpt2-large"
LAYER = 8
INTERP = "lanczos"

# Processing parameters
BATCH_SIZE = 32
VERBOSE = False
