from pathlib import Path

# Experiment configurations
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
    "wheretheressmoke_en"
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
DEFAULT_OUTPUT_PATH = Path("outputs")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATASEQ_PATH = DEFAULT_DATA_DIR / "stimuli" / "data_sequences"
DEFAULT_TEXTGRID_PATH = DEFAULT_DATA_DIR / "stimuli" / "textgrids"
DEFAULT_TRFILE_PATH = DEFAULT_DATA_DIR / "stimuli" / "trfiles"
DEFAULT_FEATURE_PATH = DEFAULT_DATA_DIR / "features"
DEFAULT_EMBEDDINGS_FILE = DEFAULT_FEATURE_PATH / "embeddings.npz"
DEFAULT_CONTEXTS_FILE = DEFAULT_FEATURE_PATH / "contexts.npz"
DEFAULT_RESPONSE_PATH = DEFAULT_DATA_DIR / "responses"
DEFAULT_MAPPER_PATH = DEFAULT_DATA_DIR / "mappers"

# Model configurations
DEFAULT_MODEL = "openai-community/gpt2-large"
DEFAULT_LAYER = 8
DEFAULT_INTERP = "lanczos"

# Processing parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_VERBOSE = False
