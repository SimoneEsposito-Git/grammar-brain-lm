"""Project-wide constants: subjects, stories, paths, and analysis settings.

Paths are relative to the working directory the pipeline is run from (the
repo root); see README.md for the expected layout of data/ and outputs/.
"""

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
IMAGES_DIR = OUTPUT_DIR / "images"
DATA_DIR = Path("data")
DATASEQ_PATH = DATA_DIR / "stimuli" / "data_sequences"
TEXTGRID_PATH = DATA_DIR / "stimuli" / "textgrids"
TRFILE_PATH = DATA_DIR / "stimuli" / "trfiles"
FEATURE_PATH = DATA_DIR / "features"
EMBEDDINGS_FILE = FEATURE_PATH / "embeddings.npz"
CONTEXTS_FILE = FEATURE_PATH / "masks.npz"
RESPONSE_PATH = DATA_DIR / "responses"
MAPPER_PATH = DATA_DIR / "mappers"

# Model configurations
MODEL = "openai-community/gpt2-large"
LAYER = 8
INTERP = "lanczos"

# Processing parameters
BATCH_SIZE = 32
VERBOSE = False

# Analysis configurations
MODALITIES = ["listening", "reading"]
MODES = [
    'baseline',
    'noun',
    'verb',
    'adj',
    'adv',
    'pron',
    'intj'
]
ROI = [
    "AC",
    "ATFP",
    "Broca",
    "EBA",
    "FEF",
    "IFSFP",
    "M1F",
    "M1H",
    "M1M",
    "OFA",
    "OPA",
    "PMvh",
    "PPA",
    "RSC",
    "S1F",
    "S1H",
    "S1M",
    "SEF",
    "SMFA",
    "SMHA",
    "cIPL",
    "hMT",
    "pIC",
    "pSTS",
    "sPMv",
]

ROI_GROUPS_ = {
    "Language & Frontal":        ["AC", "ATFP", "Broca", "IFS/FP", "M1F"],
    "Motor & Somatosensory":     ["M1H", "M1M", "S1F", "S1H", "S1M", "SMFA", "SMHA"],
    "Premotor & Supplementary":  ["PMvh", "PPA", "PSC", "SEF"],
    "Parietal & Temporal":       ["dIPL", "hMT", "pIC", "pSTS", "sMv", "OPA", "OFA", "EBA"],
}

ROI_GROUPS = {
    "Auditory Cortex":             ["AC"],
    "Early Visual Cortex":         ["V1", "V2", "V3", "V3A", "V3B", "V4"],
    "Ventral Temporal Cortex":     ["FFA", "PPA", "VO"],
    "Lower Temporal Cortex":       ["LO", "hMT", "EBA", "OFA"], 
    "Lower Parietal Cortex":       ["dIPL"],
    "Middle Parietal Cortex":      ["IPS", "V7", "OPA"],
    "Inferior Pre-Frontal Cortex": ["IFS/FP", "FO"],
    "Superior Pre-Frontal Cortex": ["SEF"],
    "Brocas Area":                 ["Broca"],
}