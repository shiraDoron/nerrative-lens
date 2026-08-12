# Narrative list
NARRATIVES = [
    "Zionist", "Resistance", "Western",
    "Russian", "Ukrainian",
    "Right-wing", "Left-wing"
]

# Dimensions
NUM_NARRATIVES = len(NARRATIVES)    # number of narratives
NUM_TOPICS = 500                    # maximum number of topics
NUM_EMOTIONS = 6                    # number of emotions

# Hyperparameters
LEARNING_RATE = 0.001   # step size
BATCH_SIZE = 16         # batch size
EPOCHS = 20             # number of iterations over the data

# Narrative-to-index mapping
NARRATIVE_TO_IDX = {name: i for i, name in enumerate(NARRATIVES)}
IDX_TO_NARRATIVE = {i: name for i, name in enumerate(NARRATIVES)}

# --- Model selection for training (train.py) ---
# Can be overridden via --model on the command line. Options:
#   "baseline_fusion" - the existing NarrativeDetector (fusion.py), unchanged
#   "sbert_only"       - Baseline 1: frozen SBERT embedding -> MLP only
#   "hybrid"           - HybridNarrativeDetector: SBERT + all engineered features
MODEL_TYPE = "baseline_fusion"
MODEL_TYPES = ("baseline_fusion", "sbert_only", "hybrid")

# --- Train/Validation/Test split ratios ---
# Shared across all three model types (same random_state, same split logic) to
# ensure a fair research comparison - every model is trained/evaluated on the exact same samples.
VAL_SIZE = 0.15
TEST_SIZE = 0.15