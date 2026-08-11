# רשימת הנרטיבים
NARRATIVES = [
    "Zionist", "Resistance", "Western",
    "Russian", "Ukrainian",
    "Right-wing", "Left-wing"
]

# מימדים
NUM_NARRATIVES = len(NARRATIVES)    # מספר הנרטיבים
NUM_TOPICS = 500                    # מספר נושאים מקסימלי
NUM_EMOTIONS = 6                    # מספר רגשות

# היפר-פרמטרים
LEARNING_RATE = 0.001   # גודל הצעד
BATCH_SIZE = 16         # גודל הקבוצה
EPOCHS = 20             # מספר האיטרציות על הנתונים

# מיפוי נרטיב לאינקדס
NARRATIVE_TO_IDX = {name: i for i, name in enumerate(NARRATIVES)}
IDX_TO_NARRATIVE = {i: name for i, name in enumerate(NARRATIVES)}

# --- בחירת מודל לאימון (train.py) ---
# ניתן לדרוס באמצעות --model בשורת הפקודה. אפשרויות:
#   "baseline_fusion" - NarrativeDetector הקיים (fusion.py), ללא שינוי
#   "sbert_only"       - Baseline 1: SBERT embedding קפוא -> MLP בלבד
#   "hybrid"           - HybridNarrativeDetector: SBERT + כל התכונות המהונדסות
MODEL_TYPE = "baseline_fusion"
MODEL_TYPES = ("baseline_fusion", "sbert_only", "hybrid")

# --- יחסי חלוקת Train/Validation/Test ---
# משותפים לשלושת סוגי המודלים (אותו random_state, אותה לוגיקת חלוקה) כדי
# להבטיח השוואה מחקרית הוגנת - כל מודל מאומן/מוערך על אותם הדגימות בדיוק.
VAL_SIZE = 0.15
TEST_SIZE = 0.15