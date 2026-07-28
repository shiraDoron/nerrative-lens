# -*- coding: utf-8 -*-
"""
ניתוח אג'נדות ומאפיינים לפי נרטיב
=================================
הסקריפט מקבל את דאטהסטי הנרטיבים (ברירת מחדל: twitter_natural_dataset.csv)
ומפיק לכל נרטיב (ציוני, רוסי, אוקראיני וכו') את האג'נדות, הערכים והמאפיינים
הרטוריים/סגנוניים שמאפיינים אותו - בלי שהתוויות האלה קיימות בדאטהסט.

השיטה (ללא תלות בספריות כבדות - רק pandas/numpy):
  1. נירמול וניקוי טקסט.
  2. פרופיל אג'נדות: לקסיקונים תמטיים (ביטחון, כלכלה, דת, זכויות אדם...)
     -> שכיחות הופעה במסמכים -> Lift מול ממוצע הקורפוס.
     מינוח: "קטגוריה" = החלוקה התמטית הקבועה מראש; "נושא" = השכיחות
     הגולמית שלה אצל נרטיב נתון (בלי קשר לייחודיות); "אג'נדה" = נושא
     שנרטיב מקדם באופן חריג ביחס לשאר הקורפוס (Lift גבוה).
  3. פרופיל רטורי/ערכי: קורבנות, גאווה, דה-לגיטימציה, איום, סולידריות, מוסר...
  4. מינוח אידאולוגי: לקסיקון נפרד שמזהה אזכור מפורש של מינוחי אידאולוגיה/
     השקפת עולם מוכרים (שמרנות, ליברליזם, סוציאליזם, ליברטריאניזם,
     לאומנות/פופוליזם, גלובליזם) - ציר ניתוח משלים לפרופיל האג'נדות.
  5. פרופיל סגנוני: אורך, האשטגים, סימני קריאה, "אנחנו" מול "הם", נתונים מספריים.
  6. מילות מפתח ייחודיות: Log-Odds-Ratio עם Informative Dirichlet Prior
     (Monroe et al. 2008) - עמיד יותר מ-TF-IDF להבדלי גודל בין קבוצות.
  7. שחקנים מרכזיים: ישויות/האשטגים/מנשנים בולטים לכל נרטיב.
  8. פלט: דוח טקסטואלי בעברית + קבצי CSV + מפת חום אופציונלית.

הרצה:
    python analyze_agendas.py
    python analyze_agendas.py --files twitter_natural_dataset.csv telegram_natural_dataset.csv
    python analyze_agendas.py --plot
"""

import argparse
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. לקסיקונים - אג'נדות תמטיות
#    '*' בסוף מילה = תחילית (למשל attack* -> attack/attacks/attacking)
# ----------------------------------------------------------------------------

AGENDA_LEXICON = {
    # פוצל מתוך "ביטחון, צבא ולוחמה" המקורית: הקטגוריה הישנה הופיעה ב-13%-55%
    # מהטקסטים בכל אחד מ-7 הנרטיבים (Lift נמוך, לא מבחינה), כי "צבא/מלחמה" הוא
    # נושא-על משותף שכולם נוגעים בו. הפיצול לארבע תת-קטגוריות ממוקדות יותר
    # (לפי אופי המסגור, לא רק הנושא) מאפשר להבחין בין נרטיב שממסגר "תקיפות
    # וסיכולים" (ציוני) לבין נרטיב שממסגר "צבא סדיר וכוחות" (אוקראיני/מערבי).
    "עימות ולוחמה - כללי": [
        "war", "warfare", "battle*", "military", "offensive", "defen*", "operation",
        "ceasefire violation",
    ],
    "צבא סדיר, כוחות ולוחמה קונבנציונלית": [
        "combat", "front line", "frontline", "army", "troop*", "soldier*", "brigade",
        "battalion", "mechanized", "infantry", "military unit*", "deployment",
        "navy", "air force", "armed forces", "occupation forces", "exercise", "drill*",
    ],
    "נשק, טילים וכלי לחימה": [
        "artiller*", "missile*", "rocket*", "drone*", "weapon*", "ammunition", "tank*", "warship",
    ],
    "תקיפות, סיכולים וחיסולים": [
        "airstrike*", "air strike", "strike*", "shelling", "bomb*", "eliminat*",
        "neutraliz*", "targeted killing", "precision strike", "took out",
        "assassinat*", "raid*", "incursion",
    ],
    "טרור ואיום קיומי": [
        "terror*", "jihad*", "extremis*", "hamas", "hezbollah", "houthi*", "isis",
        "islamic jihad", "proxy", "proxies", "militant*", "radical*", "sleeper cell",
        "existential threat", "annihilat*", "wipe out",
    ],
    "דיפלומטיה ויחסים בינלאומיים": [
        "diploma*", "negotiat*", "talks", "summit", "ambassador", "embassy", "treaty",
        "agreement", "accord*", "memorandum", "delegation", "bilateral", "multilateral",
        "united nations", "security council", "resolution", "alliance", "allies", "partner*",
        "cooperation", "dialogue", "foreign minister*", "meeting with", "ceasefire",
    ],
    "כלכלה, סחר ואנרגיה": [
        "econom*", "trade", "export*", "import*", "sanction*", "tariff*", "inflation",
        "gdp", "invest*", "investor*", "market*", "business*", "industry", "jobs",
        "employment", "wage*", "salar*", "tax*", "budget", "billion", "million dollars",
        "gas", "oil", "pipeline", "energy", "electricity", "grain", "currency", "ruble",
        "dollar", "euro", "price*", "cost of living",
    ],
    "דמוקרטיה, בחירות ומשילות": [
        "democra*", "election*", "vote*", "voting", "ballot", "parliament*", "congress",
        "senate", "referendum", "constitution*", "rule of law", "corrupt*", "governance",
        "sovereign will", "legitim*", "opposition", "civil society", "political prisoner*",
    ],
    "זכויות אדם וצדק חברתי": [
        "human right*", "civil right*", "justice", "equality", "inequality", "discriminat*",
        "racis*", "freedom of", "free speech", "dignity", "worker*", "union*", "poverty",
        "homeless*", "healthcare", "health care", "medicare", "education", "student debt",
        "billionaire*", "working class", "social justice", "welfare", "minorit*",
        "women right*", "lgbt*",
    ],
    "הומניטרי וסבל אזרחי": [
        "humanitarian", "civilian*", "children", "child", "kid*", "famine", "starv*",
        "hunger", "aid convoy", "aid", "relief", "shelter*", "displaced", "refugee*",
        "evacuat*", "hospital*", "clinic", "wounded", "injured", "casualt*", "orphan*",
        "families", "innocent*", "suffering", "genocide", "massacre*", "siege",
    ],
    "דת, מסורת וזהות": [
        "god", "allah", "lord", "holy", "sacred", "pray*", "prayer*", "faith", "islam*",
        "muslim*", "jewish", "judaism", "christian*", "church", "mosque", "synagogue",
        "quran", "bible", "torah", "ramadan", "eid", "shabbat", "passover", "hanukkah",
        "easter", "christmas", "martyr*", "blessed", "imam", "rabbi", "clerg*", "pilgrim*",
    ],
    "לאומיות, ריבונות וטריטוריה": [
        "sovereign*", "nation*", "homeland", "motherland", "fatherland", "independen*",
        "flag", "territorial integrity", "border*", "territory", "our land", "our people",
        "patriot*", "statehood", "self-determination", "annex*", "liberat*", "reunif*",
    ],
    "אנטי-אימפריאליזם וביקורת על המערב": [
        "imperialis*", "colonial*", "neocolonial*", "hegemon*", "unipolar", "multipolar",
        "western dominance", "double standard*", "arrogan*", "puppet*", "vassal",
        "apartheid", "oppress*", "exploitat*", "regime change", "interference",
        "unilateral", "russophobia", "zionist regime", "the great satan",
    ],
    "טכנולוגיה, חדשנות ומדע": [
        "technolog*", "startup*", "start-up*", "innovat*", "artificial intelligence",
        " ai ", "algorithm*", "research", "scientist*", "science", "breakthrough",
        "patent", "satellite", "space", "cyber*", "digital", "software", "engineer*",
        "medical device", "vaccine", "biotech*", "quantum",
    ],
    "תרבות, ספורט וחיי יומיום": [
        "sport*", "medal*", "champion*", "tournament", "olymp*", "football", "soccer",
        "judo", "athlete*", "festival", "music", "concert", "film", "movie", "art",
        "museum", "cuisine", "tourism", "tourist*", "beach", "spring", "holiday*",
        "celebrat*", "anniversary of the", "heritage", "tradition*",
    ],
    "תקשורת, תעמולה ודיסאינפורמציה": [
        "fake news", "propagand*", "disinformation", "misinformation", "media",
        "journalis*", "press", "censor*", "narrative", "lies", "lying", "hoax",
        "mainstream media", "fact-check*", "manipulat*", "smear", "bot farm",
    ],
    "הגירה, גבולות וביטחון פנים": [
        "migrant*", "immigra*", "illegal alien*", "asylum", "deport*", "border security",
        "border crisis", "cartel*", "trafficking", "crime", "criminal*", "police",
        "law enforcement", "gang*",
    ],
    "אקלים, אנרגיה ירוקה וסביבה": [
        "climate", "global warming", "emission*", "carbon", "renewable*", "green deal",
        "sustainab*", "environment*", "pollut*", "biodiversity", "net zero",
    ],
    "זיכרון היסטורי ועבר": [
        "history", "historic*", "memory", "remember*", "commemorat*", "anniversary",
        "holocaust", "shoah", "world war", "wwii", "nazi*", "fascis*", "ancestors",
        "generations", "legacy", "never again", "veteran*", "1948", "1967",
    ],
}

# ----------------------------------------------------------------------------
# 2. לקסיקונים - מאפיינים רטוריים / ערכיים (איך מדברים, לא על מה)
# ----------------------------------------------------------------------------

RHETORIC_LEXICON = {
    "מסגור קורבנות וסבל": [
        "victim*", "innocent*", "murder*", "killed", "slaughter*", "massacre*", "atrocit*",
        "brutal*", "cruel*", "suffer*", "mourn*", "grief", "tragedy", "heartbreak*",
        "may their memory", "rest in peace", "bloodshed", "wounded", "orphan*", "widow*",
    ],
    "גאווה, הישג והרואיות": [
        "proud", "pride", "hero*", "brave*", "courage*", "glor*", "triumph", "victor*",
        "resilien*", "strength", "strong", "historic achievement", "record", "first ever",
        "for the first time", "honor*", "honour*", "salute", "achievement", "success*",
        "world-class", "leading",
    ],
    "דה-לגיטימציה של היריב": [
        "regime", "dictator*", "tyrann*", "criminal*", "war criminal", "thug*", "puppet*",
        "evil", "barbar*", "savage*", "illegitimate", "so-called", "liar*", "lies",
        "hypocris*", "hypocrite*", "cowardl*", "rogue", "fascis*", "nazi*", "terrorist state",
        "apartheid state", "occupier*", "aggressor*", "invader*",
    ],
    "איום, אזהרה והפחדה": [
        "threat*", "danger*", "warn*", "risk*", "imminent", "escalat*", "destabiliz*",
        "nuclear weapon*", "nuclear program", "will not tolerate", "will not accept",
        "consequences", "red line", "alert", "emergency", "crisis", "if they", "before it is too late",
    ],
    "סולידריות וקריאה לפעולה": [
        "stand with", "stand together", "we must", "we call", "call on", "join us",
        "support", "solidarity", "unite*", "together we", "sign the", "take action",
        "act now", "demand*", "help us", "donate", "thank you", "grateful", "gratitude",
    ],
    "מוסר, אשמה ואחריותיות": [
        "must be held accountable", "accountab*", "war crime*", "crimes against humanity",
        "violat*", "illegal", "unlawful", "condemn*", "shameful", "shame", "unacceptable",
        "impunity", "international law", "responsib*", "justice for", "tribunal", "sanction*",
    ],
    "תקווה, עתיד ובנייה": [
        "hope*", "future", "peace*", "rebuild*", "reconstruct*", "brighter", "new beginning",
        "renewal", "progress", "vision", "dream", "prosper*", "together we will", "recovery",
        "opportunit*",
    ],
    "לעג, ציניות וסרקזם": [
        "so-called", "allegedly", "apparently", "ridiculous", "laughable", "pathetic",
        "clown*", "circus", "joke", "guess what", "surprise surprise", "sure,",
        "what a", "oops", "cardboard", "irony", "ironic",
    ],
    "אחדות פנימית וזהות \"אנחנו\"": [
        "our people", "our nation", "our soldiers", "our children", "our country",
        "our forces", "our heroes", "our values", "our future", "the people of",
        "am israel", "am yisrael", "slava ukraini", "glory to", "long live",
    ],
    "עובדתיות ונתונים": [
        "according to", "report*", "data", "statistic*", "percent", "%", "survey",
        "study", "index", "ranking", "figures", "numbers show", "analysis",
    ],
}

# ----------------------------------------------------------------------------
# 2.5 לקסיקון מינוח אידאולוגי - השקפות עולם פוליטיות מפורשות
#     בשונה מ-AGENDA_LEXICON (על מה מדברים) ו-RHETORIC_LEXICON (איך ממסגרים),
#     זהו לקסיקון שלישי שמזהה אזכור מפורש של מינוחי אידאולוגיה/השקפת עולם
#     מוכרים (שמרנות, ליברליזם, סוציאליזם...) - ציר ניתוח נפרד ומשלים.
# ----------------------------------------------------------------------------

IDEOLOGY_LEXICON = {
    "שמרנות ומסורתיות": [
        "conservative", "conservatism", "traditional values", "family values",
        "pro-life", "small government", "limited government", "law and order",
        "personal responsibility", "god-fearing", "traditional marriage",
        "judeo-christian", "founding fathers", "constitutional*",
    ],
    "ליברליזם ופרוגרסיביזם": [
        "liberal", "liberalism", "progressive", "progressivism", "inclusive",
        "inclusion", "diversity", "equity", "social justice", "identity politics",
        "cancel culture", "woke", "systemic racism", "marginalized",
    ],
    "סוציאליזם ושמאל כלכלי": [
        "socialis*", "communis*", "marxis*", "redistribut*", "wealth tax",
        "workers' rights", "class struggle", "public ownership",
        "universal healthcare", "universal basic income", "nationaliz*",
    ],
    "ליברטריאניזם וכלכלת שוק חופשי": [
        "libertarian*", "free market", "free-market", "deregulat*",
        "laissez-faire", "individual liberty", "private enterprise",
        "capitalis*", "privati*",
    ],
    "לאומנות ופופוליזם": [
        "nationalis*", "populis*", "the elite*", "establishment", "deep state",
        "ordinary people", "silent majority", "america first", "drain the swamp",
        "globalist*",
    ],
    "גלובליזם ובינלאומיות": [
        "globali*", "global governance", "international community",
        "multilateral*", "global citizen*", "one world", "international cooperation",
    ],
}

# ----------------------------------------------------------------------------
# 3. עזרי טקסט
# ----------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he her here hers herself him himself
his how i i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not
of off on once only or other ought our ours ourselves out over own same shan't she should shouldn't so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was wasn't we were weren't what when where which while who whom why with won't would
wouldn't you your yours yourself yourselves will just also new one two get got make made says said say
today day days year years time now still even much many us via amp rt http https com www co t
""".split())

# רעשי פלטפורמה / קידום עצמי - לא מייצגים אג'נדה אלא ערוץ ההפצה
PLATFORM_NOISE = set("""
follow telegram subscribe channel retweet click link links bit ly url thread
presstv novara novaramedia foxnews benshapiro mfarussia secgennato aaronbastani
defenceu zelenskyyua standwithus kremlinrussia europa int org net twitter facebook
instagram youtube tiktok whatsapp app download stream livestream
""".split())

BOILERPLATE_RE = re.compile(
    r"follow\s+press\s*tv\s+on\s+telegram|follow\s+us\s+on\s+\w+|"
    r"subscribe\s+to\s+\w+|read\s+more\s*:?|full\s+story\s*:?|click\s+here",
    re.IGNORECASE,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@(\w{2,})")
HASHTAG_RE = re.compile(r"#(\w{2,})")
WORD_RE = re.compile(r"[a-z][a-z'\-]+")
CAP_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\b")


def build_pattern(terms):
    """בונה regex יחיד לקטגוריה. '*' = תחילית."""
    parts = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if t.endswith("*"):
            body = re.escape(t[:-1]) + r"\w*"
        else:
            body = re.escape(t)
        parts.append(body)
    # \b בקצוות רק אם הקצה הוא תו מילה
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w-])", re.IGNORECASE)


AGENDA_PATTERNS = {k: build_pattern(v) for k, v in AGENDA_LEXICON.items()}
RHETORIC_PATTERNS = {k: build_pattern(v) for k, v in RHETORIC_LEXICON.items()}
IDEOLOGY_PATTERNS = {k: build_pattern(v) for k, v in IDEOLOGY_LEXICON.items()}


def clean_text(t):
    t = str(t)
    t = URL_RE.sub(" ", t)
    t = BOILERPLATE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(t):
    return [w for w in WORD_RE.findall(t.lower())
            if w not in STOPWORDS and w not in PLATFORM_NOISE and len(w) > 2]


# ----------------------------------------------------------------------------
# 4. חישובי פרופיל
# ----------------------------------------------------------------------------

def category_profile_by(df, patterns, group_col, text_col="clean"):
    """כמו category_profile, אך מאפשר לקבץ לפי כל עמודה (למשל 'account' ולא רק
    'narrative_name') - משמש לפילוח פנימי בתוך נרטיב (per-account sub-profile)."""
    rows = {}
    for cat, pat in patterns.items():
        rows[cat] = df[text_col].apply(lambda s: bool(pat.search(s)))
    hits = pd.DataFrame(rows, index=df.index)
    prof = hits.groupby(df[group_col]).mean() * 100.0
    overall = hits.mean() * 100.0
    return prof, overall


def category_profile(df, patterns, text_col="clean"):
    """אחוז המסמכים בכל נרטיב שמכילים לפחות התאמה אחת לקטגוריה."""
    return category_profile_by(df, patterns, "narrative_name", text_col=text_col)


def style_profile(df):
    txt = df["text"].astype(str)
    clean = df["clean"]
    tokens = clean.str.lower()
    n_words = clean.str.split().str.len().replace(0, np.nan)

    feats = pd.DataFrame({
        "אורך ממוצע (מילים)": n_words,
        "האשטגים לפוסט": txt.str.count(r"#\w+"),
        "אזכורי חשבונות לפוסט": txt.str.count(r"@\w+"),
        "קישורים לפוסט": txt.str.count(URL_RE),
        "סימני קריאה לפוסט": txt.str.count(r"!"),
        "סימני שאלה לפוסט": txt.str.count(r"\?"),
        "מילים בכל האותיות הגדולות (%)": txt.str.count(r"\b[A-Z]{3,}\b") / n_words * 100,
        "צפיפות מספרים (%)": clean.str.count(r"\b\d[\d,\.]*\b") / n_words * 100,
        "\"אנחנו/שלנו\" ל-100 מילים": tokens.str.count(r"\b(we|our|ours|us)\b") / n_words * 100,
        "\"הם/שלהם\" ל-100 מילים": tokens.str.count(r"\b(they|them|their|theirs)\b") / n_words * 100,
        "\"אני\" ל-100 מילים": tokens.str.count(r"\b(i|my|me|mine)\b") / n_words * 100,
    }, index=df.index)
    return feats.groupby(df["narrative_name"]).mean()


def log_odds_keywords(df, top_n=15, min_count=5):
    """Log-Odds-Ratio with Informative Dirichlet Prior (Monroe et al., 2008)."""
    counts = {}
    for nar, grp in df.groupby("narrative_name"):
        c = Counter()
        for t in grp["clean"]:
            c.update(tokenize(t))
        counts[nar] = c

    total = Counter()
    for c in counts.values():
        total.update(c)
    vocab = [w for w, n in total.items() if n >= min_count]
    a0 = sum(total[w] for w in vocab)
    alpha = {w: total[w] for w in vocab}  # prior אינפורמטיבי מהקורפוס כולו

    results = {}
    for nar, c in counts.items():
        n_i = sum(c[w] for w in vocab)
        rest = Counter()
        for other, c2 in counts.items():
            if other != nar:
                rest.update({w: c2[w] for w in vocab if c2[w]})
        n_j = sum(rest[w] for w in vocab)

        scored = []
        for w in vocab:
            y_i, y_j, a_w = c[w], rest[w], alpha[w]
            if y_i < 2:
                continue
            num_i = y_i + a_w
            num_j = y_j + a_w
            den_i = n_i + a0 - num_i
            den_j = n_j + a0 - num_j
            if den_i <= 0 or den_j <= 0:
                continue
            delta = math.log(num_i / den_i) - math.log(num_j / den_j)
            var = 1.0 / num_i + 1.0 / num_j
            scored.append((w, delta / math.sqrt(var), y_i))
        scored.sort(key=lambda x: -x[1])
        results[nar] = scored[:top_n]
    return results


def key_actors(df, top_n=12):
    """שחקנים/ישויות בולטים: שמות פרטיים באותיות גדולות + האשטגים + מנשנים."""
    per_nar, total = {}, Counter()
    for nar, grp in df.groupby("narrative_name"):
        c = Counter()
        for raw in grp["text"].astype(str):
            s = BOILERPLATE_RE.sub(" ", URL_RE.sub(" ", raw))
            for m in CAP_RE.findall(s):
                key = m.strip()
                low = key.lower()
                if low in STOPWORDS or low in PLATFORM_NOISE or len(key) < 3:
                    continue
                c[key] += 1
            for m in HASHTAG_RE.findall(s):
                if m.lower() in PLATFORM_NOISE:
                    continue
                c["#" + m] += 1
            for m in MENTION_RE.findall(s):
                c["@" + m] += 1
        per_nar[nar] = c
        total.update(c)

    out = {}
    n_docs_all = len(df)
    for nar, c in per_nar.items():
        n_docs = (df["narrative_name"] == nar).sum()
        scored = []
        for ent, cnt in c.items():
            if cnt < 3:
                continue
            share_in = cnt / n_docs
            share_all = total[ent] / n_docs_all
            lift = share_in / share_all if share_all else 0
            scored.append((ent, cnt, lift, share_in * lift))
        scored.sort(key=lambda x: -x[3])
        out[nar] = [(e, c_, l) for e, c_, l, _ in scored[:top_n]]
    return out


# ----------------------------------------------------------------------------
# 4.5 נושאים מגולים אוטומטית (LSA/SVD) - "BERTopic קל" ללא תלויות כבדות
#     BERTopic עצמו (sentence-transformers + UMAP + HDBSCAN) לא זמין בסביבה
#     המקומית (רק numpy/pandas). LSA (Latent Semantic Analysis) היא שיטת
#     topic-modeling קלאסית מבוססת TF-IDF + SVD שדורשת רק numpy, ונותנת
#     "נושאים" מגולים מהנתונים בעצמם (בניגוד ל-AGENDA_LEXICON הידני) -
#     שימושי כדי לוודא שהלקסיקון הידני לא מפספס צירים חשובים של הבדלה בין
#     נרטיבים. משתמש מחדש ב-top_distinctive/narrative_similarity/
#     distinctiveness_scores הגנריות שכבר בנויות לכל profile בעל מבנה
#     (נרטיב x קטגוריה).
# ----------------------------------------------------------------------------

def build_tfidf_matrix(df, min_df=5, max_terms=4000):
    """בונה מטריצת TF-IDF (מסמכים x מילים) מתוך df['clean'], עם pandas/numpy בלבד."""
    doc_tokens = df["clean"].apply(tokenize)
    doc_freq = Counter()
    for toks in doc_tokens:
        doc_freq.update(set(toks))

    vocab = [w for w, n in doc_freq.items() if n >= min_df]
    # אם יש יותר מדי מונחים, שומרים את הנפוצים ביותר כדי לשמור על SVD מהיר
    if len(vocab) > max_terms:
        vocab = sorted(vocab, key=lambda w: -doc_freq[w])[:max_terms]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    n_docs = len(df)
    tf = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row_i, toks in enumerate(doc_tokens):
        c = Counter(t for t in toks if t in vocab_index)
        for w, cnt in c.items():
            tf[row_i, vocab_index[w]] = cnt

    idf = np.log((n_docs + 1) / (np.array([doc_freq[w] for w in vocab]) + 1)) + 1.0
    tfidf = tf * idf
    return tfidf, vocab


def lsa_topics(tfidf, vocab, n_topics=12, top_words=10):
    """SVD חתוך (LSA) על מטריצת TF-IDF. מחזיר: מילות המפתח לכל נושא,
    והנושא הדומיננטי (עמודת |U*S| מקסימלית) לכל מסמך."""
    n_topics = min(n_topics, min(tfidf.shape) - 1) if min(tfidf.shape) > 1 else 1
    u, s, vt = np.linalg.svd(tfidf, full_matrices=False)
    u_topics = u[:, :n_topics] * s[:n_topics]

    topic_words = {}
    for t in range(n_topics):
        loadings = vt[t]
        top_idx = np.argsort(-np.abs(loadings))[:top_words]
        topic_words[t] = [vocab[i] for i in top_idx]

    dominant_topic = np.argmax(np.abs(u_topics), axis=1)
    return topic_words, dominant_topic


def data_driven_topic_profile(df, n_topics=12, min_df=5, max_terms=4000, top_words=10):
    """מפעיל את כל צינור ה-LSA ומחזיר profile/overall בפורמט זהה ל-category_profile,
    כדי לאפשר שימוש חוזר ב-top_distinctive/narrative_similarity/distinctiveness_scores."""
    tfidf, vocab = build_tfidf_matrix(df, min_df=min_df, max_terms=max_terms)
    topic_words, dominant_topic = lsa_topics(tfidf, vocab, n_topics=n_topics, top_words=top_words)

    labels = {t: " / ".join(words[:4]) for t, words in topic_words.items()}
    assigned = pd.Series([labels[t] for t in dominant_topic], index=df.index)

    hits = pd.get_dummies(assigned) * 100.0
    profile = hits.groupby(df["narrative_name"]).mean()
    overall = hits.mean()
    return profile, overall, topic_words, labels


def top_distinctive(profile, overall, k=5, min_prevalence=3.0, min_lift=1.1):
    """מחזיר לכל נרטיב את הקטגוריות עם ה-Lift הגבוה ביותר מול הקורפוס.
    מוצגות רק קטגוריות שבאמת בולטות (מעל הממוצע); אם אין מספיק - הרשימה קצרה יותר."""
    lift = profile.divide(overall.replace(0, np.nan), axis=1)
    out = {}
    for nar in profile.index:
        row = pd.DataFrame({"prev": profile.loc[nar], "lift": lift.loc[nar]})
        row = row[(row["prev"] >= min_prevalence) & (row["lift"] >= min_lift)]
        out[nar] = row.sort_values("lift", ascending=False).head(k)
    return out, lift


def narrative_similarity(profile):
    """מטריצת Cosine-Similarity בין וקטורי האג'נדות (%) של כל זוג נרטיבים.
    1.0 = פרופיל אג'נדות זהה (חפיפה מלאה, לא מובחן); 0.0 = פרופילים אורתוגונליים
    (מובחנות מלאה). משמש למדידת "מיקוד" הנרטיבים אחד מול השני."""
    mat = profile.fillna(0.0).values.astype(float)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized = mat / norm
    sim = normalized @ normalized.T
    return pd.DataFrame(sim, index=profile.index, columns=profile.index)


def distinctiveness_scores(top_agenda_dict):
    """ציון מיקוד לכל נרטיב = ה-Lift הממוצע של קטגוריות האג'נדה המובילות שלו
    (מתוך top_distinctive). ציון גבוה = הנרטיב "בעלים" של הנושאים שהוא מקדם
    ופחות חופף לנרטיבים אחרים; ציון קרוב ל-1.0 = הנרטיב לא ממוקד ומזוהה
    בעיקר עם נושאים גנריים שכולם עוסקים בהם."""
    scores = {}
    for nar, tbl in top_agenda_dict.items():
        scores[nar] = float(tbl["lift"].mean()) if len(tbl) else 0.0
    return pd.Series(scores, name="focus_score")


def shared_agendas(profile, overall, lift, min_prevalence=3.0, top_k=6):
    """מדרג קטגוריות לפי מידת ה"שיתוף" שלהן בין הנרטיבים, במקום סף בינארי
    (במדגם עם 7 נרטיבים על 3 קונפליקטים שונים, כמעט אין קטגוריה עם Lift זהה
    לגמרי אצל כולם - לכן מדרגים במקום לסנן). קטגוריה "משותפת" = מופיעה בנפח
    משמעותי אצל *כל* הנרטיבים (>= min_prevalence%), ו"יחס Lift" (מקס/מין בין
    הנרטיבים) קרוב ל-1.0 - כלומר אף נרטיב לא "בעלים" שלה יותר מהאחרים.
    בניגוד ל-top_distinctive שמראה את מה שמבדיל, זו מראה את הרקע המשותף,
    מדורג מהמשותף ביותר (יחס Lift נמוך) להכי פחות משותף."""
    rows = []
    for cat in profile.columns:
        prev = profile[cat]
        lif = lift[cat]
        if (prev >= min_prevalence).all():
            lo, hi = float(lif.min()), float(lif.max())
            ratio = hi / lo if lo > 0 else float("inf")
            rows.append({
                "קטגוריה": cat,
                "שכיחות ממוצעת (%)": float(prev.mean()),
                "שכיחות מינ' (%)": float(prev.min()),
                "שכיחות מקס' (%)": float(prev.max()),
                "Lift ממוצע": float(lif.mean()),
                "יחס Lift (מקס/מין)": ratio,
            })
    out = pd.DataFrame(rows, columns=["קטגוריה", "שכיחות ממוצעת (%)", "שכיחות מינ' (%)",
                                      "שכיחות מקס' (%)", "Lift ממוצע", "יחס Lift (מקס/מין)"])
    if len(out):
        out = out.sort_values("יחס Lift (מקס/מין)").reset_index(drop=True)
        if top_k:
            out = out.head(top_k)
    return out


def intra_narrative_diversity(df, patterns, min_docs_per_account=30, min_accounts=2):
    """מפלח כל נרטיב לפי חשבונות המקור שכבר קיימים בדאטה (בלי לאסוף נתונים
    חדשים!) - כל נרטיב כבר מורכב מכמה חשבונות (למשל "ציוני" = Israel/IDF/
    StandWithUs/AIPAC). בונה פרופיל אג'נדות פר-חשבון (ביחס לממוצע הנרטיב
    עצמו, לא לקורפוס כולו) ומודד קוסינוס-דמיון פנימי בין זוגות החשבונות.
    דמיון פנימי נמוך = הנרטיב הוא בעצם "מקשה" לא אחידה שמורכבת מכמה קולות
    שונים מאוד (למשל קול ממשלתי-רשמי מול קול אקטיביסטי/תקשורתי) - כלומר יש
    בתוכו פוטנציאל לפיצול לתת-נרטיבים ממוקדים יותר (בדומה לאיך שכבר הפרדנו
    Right-wing מ-Left-wing באמריקאית). דמיון גבוה = הנרטיב הומוגני יחסית.
    מחזיר dict: narrative -> {n_accounts, avg_internal_similarity, profile,
    overall, similarity, most_divergent_pair}."""
    results = {}
    for nar, grp in df.groupby("narrative_name"):
        counts = grp["account"].value_counts()
        valid = counts[counts >= min_docs_per_account].index
        sub = grp[grp["account"].isin(valid)]
        n_acc = sub["account"].nunique()
        if n_acc < min_accounts:
            results[nar] = {"n_accounts": n_acc, "avg_internal_similarity": np.nan,
                            "profile": None, "overall": None, "similarity": None,
                            "most_divergent_pair": None}
            continue
        acc_prof, acc_overall = category_profile_by(sub, patterns, "account")
        sim = narrative_similarity(acc_prof)
        mask = ~np.eye(len(sim), dtype=bool)
        off_diag = sim.values[mask]
        avg_sim = float(off_diag.mean()) if len(off_diag) else np.nan
        most_divergent = sim.where(mask).stack().idxmin() if len(off_diag) else None
        results[nar] = {
            "n_accounts": n_acc,
            "avg_internal_similarity": avg_sim,
            "profile": acc_prof,
            "overall": acc_overall,
            "similarity": sim,
            "most_divergent_pair": most_divergent,
        }
    return results


# ----------------------------------------------------------------------------
# 5. הפקת הדוח בעברית
# ----------------------------------------------------------------------------

NARRATIVE_HE = {
    "Zionist": "ציוני",
    "Resistance": "ציר ההתנגדות",
    "Western": "מערבי",
    "Russian": "רוסי",
    "Ukrainian": "אוקראיני",
    "Right-wing": "ימני (שמרני-אמריקאי)",
    "Left-wing": "שמאלי (פרוגרסיבי)",
}


def build_report(df, agenda_prof, agenda_overall, rhet_prof, rhet_overall,
                 style_prof, keywords, actors, auto_topics=None, account_diversity=None,
                 ideology_prof=None, ideology_overall=None):
    top_agenda, agenda_lift = top_distinctive(agenda_prof, agenda_overall, k=5)
    top_rhet, rhet_lift = top_distinctive(rhet_prof, rhet_overall, k=4)
    top_ideology, ideology_lift = (None, None)
    if ideology_prof is not None:
        top_ideology, ideology_lift = top_distinctive(ideology_prof, ideology_overall, k=3,
                                                       min_prevalence=1.0, min_lift=1.1)
    focus = distinctiveness_scores(top_agenda).sort_values(ascending=False)
    sim = narrative_similarity(agenda_prof)
    off_diag = sim.values[~np.eye(len(sim), dtype=bool)]
    avg_overlap = float(off_diag.mean()) if len(off_diag) else 0.0
    shared = shared_agendas(agenda_prof, agenda_overall, agenda_lift)

    L = []
    add = L.append
    add("=" * 78)
    add("ניתוח נרטיבים: אג'נדות, ערכים ומאפיינים")
    add("=" * 78)
    add(f"סה\"כ טקסטים שנותחו: {len(df)}   |   מספר נרטיבים: {df['narrative_name'].nunique()}")
    add("")
    add("שיטה: לקסיקונים תמטיים ורטוריים -> שכיחות במסמכים -> Lift מול ממוצע הקורפוס;")
    add("      מילות מפתח ייחודיות לפי Log-Odds-Ratio עם Dirichlet Prior.")
    add("      Lift 1.0 = כמו הממוצע. Lift 2.0 = שכיח פי 2 מהממוצע בקורפוס.")
    add("")
    add("מינוח ניתוח:")
    add("  'קטגוריה' = חלוקה תמטית קבועה מראש בלקסיקון (למשל 'כלכלה, סחר ואנרגיה').")
    add("  'נושא' = שכיחות הופעת קטגוריה בפועל אצל נרטיב מסוים, בלי קשר לייחודיות (אולי נפוץ אצל כולם).")
    add("  'אג'נדה' = נושא שנרטיב מקדם באופן חריג יחסית לשאר הקורפוס (Lift גבוה) -")
    add("  כלומר, נבחר/מודגש יותר מהרגיל בקורפוס (ברוח Agenda-Setting Theory).")
    add("")
    add("─" * 78)
    add("ציון מיקוד לנרטיב (Lift ממוצע של האג'נדות המובילות - גבוה יותר = ממוקד יותר):")
    add("─" * 78)
    for nar, score in focus.items():
        he = NARRATIVE_HE.get(nar, nar)
        add(f"      • {he:<24} ({nar:<10})   {score:.2f}x")
    add("")
    add(f"חפיפה ממוצעת בין נרטיבים (Cosine Similarity, 0=מובחן לגמרי, 1=חופף לגמרי): {avg_overlap:.2f}")
    most_overlap = sim.where(~np.eye(len(sim), dtype=bool)).stack().idxmax()
    add(f"הזוג עם החפיפה הגבוהה ביותר: {most_overlap[0]} <-> {most_overlap[1]} "
        f"({sim.loc[most_overlap]:.2f})")
    add("")

    add("─" * 78)
    add("נושאים משותפים לכל הנרטיבים (מדורג מהמשותף ביותר להכי פחות - אלו אינם 'אג'נדות' של שום נרטיב,")
    add("יחס Lift קרוב ל-1.0 = אף נרטיב לא 'בעלים' שלו יותר מהאחרים):")
    add("─" * 78)
    if len(shared):
        for _, r in shared.iterrows():
            cat = r["קטגוריה"]
            avg_prev = r["שכיחות ממוצעת (%)"]
            min_prev = r["שכיחות מינ' (%)"]
            max_prev = r["שכיחות מקס' (%)"]
            avg_lift = r["Lift ממוצע"]
            lift_ratio = r["יחס Lift (מקס/מין)"]
            add(f"      • {cat:<38} ממוצע {avg_prev:5.1f}%   "
                f"(טווח {min_prev:.1f}%-{max_prev:.1f}%,   "
                f"Lift ממוצע ×{avg_lift:.2f}, יחס Lift מקס/מין {lift_ratio:.2f})")
    else:
        add("      (אין קטגוריה עם נוכחות מספקת בכל הנרטיבים גם יחד)")
    add("")

    for nar in agenda_prof.index:
        he = NARRATIVE_HE.get(nar, nar)
        sub = df[df["narrative_name"] == nar]
        accounts = ", ".join(sub["account"].dropna().unique()[:8]) if "account" in sub else "-"
        add("")
        add("─" * 78)
        add(f"■ נרטיב: {he}  ({nar})   |   {len(sub)} טקסטים")
        if accounts and accounts != "-":
            add(f"  מקורות: {accounts}")
        add("─" * 78)

        add("")
        add("  ▸ אג'נדות מובילות (הנושאים שהנרטיב מקדם באופן ייחודי):")
        for cat, r in top_agenda[nar].iterrows():
            add(f"      • {cat:<38} {r['prev']:5.1f}% מהטקסטים   (Lift ×{r['lift']:.2f})")

        add("")
        add("  ▸ נושאים בולטים בנפח (הכי נפוצות אצלו, בלי קשר לייחודיות - לא בהכרח 'אג'נדה'):")
        for cat, val in agenda_prof.loc[nar].sort_values(ascending=False).head(4).items():
            add(f"      • {cat:<38} {val:5.1f}% מהטקסטים")

        add("")
        add("  ▸ מאפיינים רטוריים וערכיים (לפי שכיחות, עם ייחודיות):")
        rhet_rows = pd.DataFrame({"prev": rhet_prof.loc[nar], "lift": rhet_lift.loc[nar]}) \
            .sort_values("prev", ascending=False).head(5)
        for cat, r in rhet_rows.iterrows():
            flag = " ★" if r["lift"] >= 1.3 else ""
            add(f"      • {cat:<38} {r['prev']:5.1f}% מהטקסטים   (Lift ×{r['lift']:.2f}){flag}")
        if top_ideology is not None:
            ideo_rows = top_ideology[nar]
            add("")
            add("  ▸ מינוח אידאולוגי (השקפות עולם מפורשות שמוזכרות בטקסט, לא נושא/אג'נדה):")
            if len(ideo_rows):
                for cat, r in ideo_rows.iterrows():
                    add(f"      • {cat:<38} {r['prev']:5.1f}% מהטקסטים   (Lift ×{r['lift']:.2f})")
            else:
                add("      (אין אזכור משמעותי למינוח אידיאולוגי עצמאי - הנרטיב מסתפק בעיקר על מסגור עקיף, לא על תיוג עצמו)")
        add("")
        add("  ▸ סגנון (המאפיינים שבהם הנרטיב בולט ביחס לאחרים):")
        s = style_prof.loc[nar]
        rank = style_prof.rank(ascending=False)
        notable = rank.loc[nar][s > 0].sort_values().head(4)
        for feat in notable.index:
            add(f"      • {feat:<38} {s[feat]:6.2f}   (מקום {int(notable[feat])} מבין {len(style_prof)})")

        add("")
        add("  ▸ מילות מפתח ייחודיות (Log-Odds z-score):")
        kw = ", ".join(f"{w} ({z:.1f})" for w, z, _ in keywords[nar][:14])
        add(f"      {kw}")

        add("")
        add("  ▸ שחקנים / ישויות מרכזיים:")
        ac = ", ".join(f"{e} ×{c}" for e, c, _ in actors[nar][:12])
        add(f"      {ac}")

        if account_diversity is not None:
            ad = account_diversity.get(nar)
            if ad and ad["n_accounts"] >= 2:
                add("")
                add("  ▸ פילוג פנימי בין חשבונות המקור (מתוך הנתונים הקיימים, ללא איסוף חדש):")
                add(f"      חשבונות שנותחו: {ad['n_accounts']}   |   דמיון פנימי ממוצע: "
                    f"{ad['avg_internal_similarity']:.2f}   (1.0=כל החשבונות נשמעים אותו דבר, "
                    f"0.0=קולות שונים לגמרי בתוך אותו נרטיב)")
                if ad["most_divergent_pair"]:
                    a1, a2 = ad["most_divergent_pair"]
                    pair_sim = ad["similarity"].loc[a1, a2]
                    add(f"      הזוג הכי מפוצל בתוך הנרטיב: {a1} <-> {a2}   (דמיון {pair_sim:.2f})")
                acc_top, _ = top_distinctive(ad["profile"], ad["overall"], k=3,
                                             min_prevalence=2.0, min_lift=1.15)
                for acc in ad["profile"].index:
                    row = acc_top[acc]
                    if len(row):
                        cats = ", ".join(f"{c} (×{r['lift']:.1f})" for c, r in row.iterrows())
                        add(f"        • {acc:<20} מקדם יחסית לשאר הנרטיב: {cats}")

        add("")
        add("  ▸ תמצית:")
        ag_list = list(top_agenda[nar].index[:3]) or list(agenda_prof.loc[nar].nlargest(3).index)
        rh_list = [c for c, r in rhet_rows.iterrows() if r["lift"] >= 1.2][:2] or list(rhet_rows.index[:2])
        add(f"      נרטיב {he} מקדם בעיקר: {' | '.join(ag_list)}.")
        add(f"      הטון הדומיננטי: {' + '.join(rh_list)}.")
        add(f"      מסגור מרכזי סביב: {', '.join(e for e, _, _ in actors[nar][:5])}.")

    add("")
    add("=" * 78)
    add("טבלת השוואה: נושאים / קטגוריות תוכן (% מהטקסטים בכל נרטיב - לא מסונן לייחודיות)")
    add("=" * 78)
    add(agenda_prof.round(1).T.to_string())
    add("")
    add("=" * 78)
    add("טבלת השוואה: מאפיינים רטוריים (% מהטקסטים)")
    add("=" * 78)
    add(rhet_prof.round(1).T.to_string())

    if ideology_prof is not None:
        add("")
        add("=" * 78)
        add("טבלת השוואה: מינוח אידאולוגי (% מהטקסטים)")
        add("=" * 78)
        add(ideology_prof.round(1).T.to_string())

    if auto_topics is not None:
        auto_prof, auto_overall, topic_words, labels = auto_topics
        auto_top, auto_lift = top_distinctive(auto_prof, auto_overall, k=3, min_prevalence=1.0, min_lift=1.05)
        auto_focus = distinctiveness_scores(auto_top).sort_values(ascending=False)
        auto_sim = narrative_similarity(auto_prof)
        add("")
        add("=" * 78)
        add("נושאים שהתגלו אוטומטית מהנתונים (LSA/SVD - ללא לקסיקון ידני)")
        add("=" * 78)
        add("השוואה בין מה שהלקסיקון הידני 'חשב' שחשוב לבין הצירים שבאמת בולטים")
        add("בנתונים עצמם. אם נרטיב מסוים מקבל כאן נושא דומיננטי שלא מופיע כלל")
        add("ב-AGENDA_LEXICON - ייתכן שכדאי להוסיף לו קטגוריה ייעודית.")
        add("")
        for nar, score in auto_focus.items():
            he = NARRATIVE_HE.get(nar, nar)
            add(f"      • {he:<24} ({nar:<10})   ציון מיקוד: {score:.2f}x")
        add("")
        for nar in auto_prof.index:
            he = NARRATIVE_HE.get(nar, nar)
            add(f"  ▸ {he} ({nar}):")
            for cat, r in auto_top[nar].iterrows():
                add(f"      • {cat:<45} {r['prev']:5.1f}% מהטקסטים   (Lift ×{r['lift']:.2f})")
        add("")
        add("מטריצת חפיפה בין נרטיבים (על בסיס הנושאים האוטומטיים):")
        add(auto_sim.round(2).to_string())

    return "\n".join(L)


def plot_heatmap(profile, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = profile.T
    fig, ax = plt.subplots(figsize=(1.3 * len(data.columns) + 6, 0.42 * len(data.index) + 3))
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([i[::-1] for i in data.index])  # היפוך ידני ל-RTL
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title(title[::-1])
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. main
# ----------------------------------------------------------------------------

def main():
    # קונסולת Windows (cp1255/cp1252) לא תמיד יודעת להדפיס Unicode מלא (עברית
    # וסימנים כמו ×/✓) ומקריסה את print() עם UnicodeEncodeError. מעדכנים את
    # קידוד הפלט ל-UTF-8 עם fallback להחלפת תווים בעייתיים במקום קריסה.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="ניתוח אג'נדות ומאפיינים לפי נרטיב")
    ap.add_argument("--files", nargs="+", default=["twitter_natural_dataset.csv"])
    ap.add_argument("--out-prefix", default="narrative_agendas")
    ap.add_argument("--plot", action="store_true", help="שמירת מפות חום PNG")
    ap.add_argument("--min-len", type=int, default=15, help="אורך מינימלי בתווים")
    ap.add_argument("--auto-topics", type=int, default=0, metavar="N",
                    help="מגלה N נושאים אוטומטית מהנתונים (LSA/SVD, ללא BERTopic/torch) "
                         "להשוואה מול הלקסיקון הידני. 0 = כבוי (ברירת מחדל).")
    ap.add_argument("--min-docs-per-account", type=int, default=30,
                    help="מספר מסמכים מינימלי לכל חשבון-מקור כדי שייכלל בפילוג "
                         "הפנימי של הנרטיב (פילוג לפי חשבונות מקור).")
    args = ap.parse_args()

    frames = []
    for f in args.files:
        d = pd.read_csv(f)
        d["source_file"] = os.path.basename(f)
        if "account" not in d.columns:
            d["account"] = np.nan
        frames.append(d[["text", "label", "narrative_name", "account", "source_file"]])
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["text", "narrative_name"])
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() >= args.min_len]
    df = df.drop_duplicates(subset=["clean"]).reset_index(drop=True)
    print(f"[i] נטענו {len(df)} טקסטים ייחודיים מתוך {len(args.files)} קבצים.")

    agenda_prof, agenda_overall = category_profile(df, AGENDA_PATTERNS)
    rhet_prof, rhet_overall = category_profile(df, RHETORIC_PATTERNS)
    ideology_prof, ideology_overall = category_profile(df, IDEOLOGY_PATTERNS)
    style_prof = style_profile(df)
    keywords = log_odds_keywords(df)
    actors = key_actors(df)
    top_agenda_for_csv, agenda_lift_for_csv = top_distinctive(agenda_prof, agenda_overall, k=5)
    focus_scores = distinctiveness_scores(top_agenda_for_csv)
    similarity = narrative_similarity(agenda_prof)
    shared = shared_agendas(agenda_prof, agenda_overall, agenda_lift_for_csv)
    print("[i] בודק פילוג פנימי בין חשבונות המקור בתוך כל נרטיב...")
    account_diversity = intra_narrative_diversity(
        df, AGENDA_PATTERNS, min_docs_per_account=args.min_docs_per_account)

    auto_topics = None
    if args.auto_topics > 0:
        print(f"[i] מגלה {args.auto_topics} נושאים אוטומטית (LSA/SVD)...")
        auto_prof, auto_overall, topic_words, labels = data_driven_topic_profile(
            df, n_topics=args.auto_topics)
        auto_topics = (auto_prof, auto_overall, topic_words, labels)

    report = build_report(df, agenda_prof, agenda_overall, rhet_prof, rhet_overall,
                          style_prof, keywords, actors, auto_topics=auto_topics,
                          account_diversity=account_diversity,
                          ideology_prof=ideology_prof, ideology_overall=ideology_overall)

    txt_path = f"{args.out_prefix}_report.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    agenda_prof.round(2).to_csv(f"{args.out_prefix}_agendas.csv", encoding="utf-8-sig")
    rhet_prof.round(2).to_csv(f"{args.out_prefix}_rhetoric.csv", encoding="utf-8-sig")
    ideology_prof.round(2).to_csv(f"{args.out_prefix}_ideology.csv", encoding="utf-8-sig")
    style_prof.round(3).to_csv(f"{args.out_prefix}_style.csv", encoding="utf-8-sig")
    pd.DataFrame({n: [f"{w} ({z:.1f})" for w, z, _ in k] for n, k in keywords.items()}) \
        .to_csv(f"{args.out_prefix}_keywords.csv", index=False, encoding="utf-8-sig")
    focus_scores.round(3).to_csv(f"{args.out_prefix}_focus_scores.csv", encoding="utf-8-sig")
    similarity.round(3).to_csv(f"{args.out_prefix}_similarity.csv", encoding="utf-8-sig")
    shared.round(3).to_csv(f"{args.out_prefix}_shared_agendas.csv", index=False, encoding="utf-8-sig")
    diversity_rows = [{"narrative_name": nar, "n_accounts": ad["n_accounts"],
                       "avg_internal_similarity": ad["avg_internal_similarity"]}
                      for nar, ad in account_diversity.items()]
    pd.DataFrame(diversity_rows).round(3).to_csv(
        f"{args.out_prefix}_intra_narrative_diversity.csv", index=False, encoding="utf-8-sig")
    for nar, ad in account_diversity.items():
        if ad["profile"] is not None:
            safe_nar = re.sub(r"[^\w\-]+", "_", nar)
            ad["profile"].round(2).to_csv(
                f"{args.out_prefix}_accounts_{safe_nar}.csv", encoding="utf-8-sig")
    if auto_topics is not None:
        auto_prof, auto_overall, topic_words, labels = auto_topics
        auto_prof.round(2).to_csv(f"{args.out_prefix}_auto_topics.csv", encoding="utf-8-sig")
        pd.DataFrame({"topic": list(topic_words.keys()),
                     "label": [labels[t] for t in topic_words],
                     "top_words": [", ".join(w) for w in topic_words.values()]}) \
            .to_csv(f"{args.out_prefix}_auto_topic_words.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_heatmap(agenda_prof, "אג'נדות לפי נרטיב (%)", f"{args.out_prefix}_agendas.png")
        plot_heatmap(rhet_prof, "מאפיינים רטוריים לפי נרטיב (%)", f"{args.out_prefix}_rhetoric.png")

    print(report)
    print(f"\n[✓] הדוח נשמר ב-{txt_path} (+ קבצי CSV).")


if __name__ == "__main__":
    main()
