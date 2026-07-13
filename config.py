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