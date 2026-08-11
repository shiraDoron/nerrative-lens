# Narrative Detection & Profiling Pipeline

A research/thesis project that classifies text into 7 political/geopolitical **narratives** —
`Zionist`, `Resistance`, `Western`, `Russian`, `Ukrainian`, `Right-wing`, `Left-wing` — using a
hybrid pipeline of NLP feature extractors fused with a learned classification network, plus a
separate unsupervised tool for profiling each narrative's agendas, rhetoric, and ideology.

## Project Structure

```
final_project/
├── src/                          # all Python source modules
│   ├── config.py                 # narrative list, hyperparameters
│   ├── train.py                  # trains the hybrid NarrativeDetector model
│   ├── train_topics.py           # fits the BERTopic topic model
│   ├── llm_topic_refiner.py      # improves BERTopic topic labels via Gemini (Colab only)
│   ├── analyze_agendas.py        # unsupervised agenda/rhetoric/ideology profiling (local)
│   ├── check_agenda_coverage.py  # verifies dataset has enough agenda diversity per narrative
│   ├── translate_datasets.py     # translates non-English rows to English in place
│   ├── build_ai_dataset.py       # generates synthetic narrative text via Gemini
│   ├── emotion.py                # emotion + passive/active voice feature extractor
│   ├── fusion.py                 # NarrativeDetector: fuses all feature layers
│   ├── ner.py                    # named-entity feature extractor
│   ├── reliability.py            # fake-news / subjectivity confidence factor
│   ├── srl.py                    # semantic role labeling (reason/purpose) extractor
│   ├── stance.py                 # BERTopic-based topic layer
│   ├── build_twitter_dataset.py  # Selenium-based X/Twitter scraper
│   └── build_telegram_dataset.py # Telethon-based Telegram scraper
├── data/
│   ├── raw/                      # source datasets (twitter/telegram/gemini/gpt CSVs)
│   └── cache/                    # cached extracted features, shared vocabulary
├── models/
│   ├── best_narrative_model_hybrid.pth   # trained model weights
│   └── saved_topic_model/                # trained BERTopic model
├── reports/                       # generated analysis reports (CSV/PNG/TXT)
├── experiments/
│   ├── with_mlp/                 # alternate MLP-based fusion checkpoint (for comparison)
│   └── with_stance_model/        # alternate fusion variant with a stance dimension
└── narrative_research_session.session  # local Telethon auth session (gitignored)
```

## Running the Scripts

All scripts that live in `src/` must be run **from the repository root** (not from inside
`src/`), since their internal file paths (e.g. `data/raw/...`, `models/...`, `reports/...`) are
relative to the project root:

```bash
python src/analyze_agendas.py
python src/check_agenda_coverage.py
python src/train.py                       # trains config.py's MODEL_TYPE (default: baseline_fusion)
python src/train.py --model baseline_fusion  # existing NarrativeDetector (unchanged)
python src/train.py --model sbert_only       # Baseline 1: frozen SBERT embedding -> MLP
python src/train.py --model hybrid           # HybridNarrativeDetector (SBERT + engineered features)

# Generalization splits (any --model works with any --split):
python src/train.py --model hybrid --split random                                   # default: random train/val/test split
python src/train.py --model hybrid --split leave_one_topic --held-out-topic 12      # Leave-One-Topic-Out (LOTO)
python src/train.py --model hybrid --split leave_one_author --held-out-author IDF   # Leave-One-Author-Out (LOAO)
python src/train_topics.py
python src/build_twitter_dataset.py
python src/build_telegram_dataset.py
```

## Environment Variables (Secrets)

No secrets are hardcoded in the source code. Set the following environment variables before
running the relevant script:

| Variable | Used by | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | `build_ai_dataset.py`, `llm_topic_refiner.py` | Google Gemini API access |
| `TWITTER_AUTH_TOKEN` | `build_twitter_dataset.py` | X/Twitter `auth_token` cookie for scraping |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | `build_telegram_dataset.py` | Telegram API credentials |

PowerShell example:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
$env:TWITTER_AUTH_TOKEN = "<your-auth-token-cookie>"
$env:TELEGRAM_API_ID = "<id>"
$env:TELEGRAM_API_HASH = "<hash>"
```

## Dependencies

Locally installed: `pandas`, `numpy`, `matplotlib`, `selenium`, `webdriver_manager` (Microsoft
Edge is used as the scraping browser).

The following are required only for the heavier training/feature-extraction pipeline
(`train.py`, `fusion.py`, `ner.py`, `srl.py`, `emotion.py`, `reliability.py`, `stance.py`,
`train_topics.py`, `llm_topic_refiner.py`) and are intended to run in a Colab/GPU environment:
`torch`, `transformers`, `spacy` (`en_core_web_trf`), `bertopic`, `scikit-learn`,
`deep-translator`, `telethon`, `google-genai`, `sentence-transformers` (used by
`HybridNarrativeDetector` in `fusion.py`).

## Pipeline Overview

1. **Data collection**: `build_twitter_dataset.py` / `build_telegram_dataset.py` scrape
   narrative-labeled posts from a fixed set of accounts/channels per narrative;
   `build_ai_dataset.py` supplements this with synthetic Gemini-generated text.
2. **Preprocessing**: `translate_datasets.py` translates non-English text to English in place.
3. **Training**: `train_topics.py` fits a BERTopic topic model; `train.py` extracts features
   (NER, SRL, emotion/agency, topic/stance, reliability) for every sample, fuses them via
   `fusion.py`'s `NarrativeDetector`, and trains the classifier with early stopping.
   `train.py` can train any of three models (`--model baseline_fusion|sbert_only|hybrid`,
   see "Model Comparison" below) using identical train/validation/test splits for a fair
   research comparison.
4. **Analysis**: `analyze_agendas.py` and `check_agenda_coverage.py` provide a lightweight,
   dependency-free (pandas/numpy only) profiling of each narrative's agendas, rhetoric,
   ideology, and per-account internal diversity — independent of the trained model.

## Model Architecture (`fusion.py`)

Each input text passes through several frozen feature extractors, each producing a
narrative-oriented vector, which are combined by a learned weighted-sum fusion network:

- **NER** (`ner.py`) → entity-based narrative signal
- **SRL** (`srl.py`) → reason/purpose clause signal
- **Emotion + agency** (`emotion.py`) → emotion classification + passive/active voice
- **Topic/stance** (`stance.py`) → BERTopic topic assignment
- **Reliability** (`reliability.py`) → fake-news/subjectivity confidence multiplier

The fusion network learns per-module importance weights, printed after training for
interpretability.

### Hybrid v1 (`HybridNarrativeDetector`)

An additional model class in `fusion.py`, built as a first step toward a stronger,
end-to-end architecture without discarding any of the existing engineered features.
It concatenates a shared SBERT sentence embedding (`sentence-transformers/all-MiniLM-L6-v2`,
frozen) with all the existing engineered feature vectors — NER, SRL, emotion/agency,
topic/stance, reliability — plus two new lexicon-based feature vectors (agenda and
ideology, reusing `AGENDA_PATTERNS`/`IDEOLOGY_PATTERNS` from `analyze_agendas.py` via the
new `AgendaIdeologyFeatureExtractor`), then passes the combined vector through an MLP to
predict the narrative.

### Model Comparison (`train.py --model ...`)

`train.py` can train and evaluate three models, selected via `--model` (or `config.py`'s
`MODEL_TYPE`) — no manual code edits required:

| `--model` value | Class | Description |
|---|---|---|
| `baseline_fusion` | `NarrativeDetector` | Existing linear-fusion model (unchanged) |
| `sbert_only` | `SBERTOnlyDetector` | Baseline 1: frozen SBERT embedding → MLP only |
| `hybrid` | `HybridNarrativeDetector` | Proposed Hybrid: SBERT + all engineered features → MLP |

All three models are trained/evaluated on **identical train/validation/test splits**
(`config.py`'s `VAL_SIZE`/`TEST_SIZE`, fixed `random_state=42`) for a fair comparison. Each
model's checkpoint (`models/best_model_*.pth`) is selected by **validation Macro-F1** (not
loss). Accuracy, Macro-F1, Macro-Precision and Macro-Recall for the validation and test splits
of every run are accumulated in `reports/model_comparison_results.json`.

Interpretability for the Hybrid model is intended to come from **ablation studies** (removing
one feature group at a time and measuring the Macro-F1 drop) rather than learned fusion
weights — this is scaffolded via `evaluate()`'s `features_labels` parameter but not yet
implemented (see the "future extension points" comment block at the bottom of `train.py`).

### Generalization evaluation (`--split ...`)

Every dataset row is tagged with two provenance columns during loading (`load_raw_data()` in
`train.py`): `dataset_source` (`gemini`/`gpt`/`twitter`/`telegram`) and `author_source` (the
specific account/channel — for `twitter`/`telegram` this is the real `account` column already
written by the scrapers; for the synthetic `gemini`/`gpt` datasets, which have no real per-row
author, a placeholder value like `gemini_synthetic` is used instead). `--split` selects how
train/validation/test are built from these:

| `--split` value | Behavior | Extra flag required |
|---|---|---|
| `random` (default) | Ordinary random split (`config.py`'s `VAL_SIZE`/`TEST_SIZE`, `random_state=42`) | — |
| `leave_one_topic` | All samples of one BERTopic topic id go entirely to test; rest split train/val | `--held-out-topic <topic_id>` |
| `leave_one_author` | All samples of one account/channel (`author_source`) go entirely to test; rest split train/val | `--held-out-author <name>` |

For the two specialized modes, `train.py` automatically runs `verify_no_leakage()` to assert
the held-out topic/author never also appears in train or validation, and saves a
`reports/split_summary_<model>_<run>.json` file with per-split narrative counts and distinct
`dataset_source`/`author_source` counts. Cache (`data/cache/`) and checkpoint (`models/`) files
are automatically namespaced per split mode + held-out value, so a `random`-split run never
collides with a `leave_one_topic`/`leave_one_author` run of the same model.

**Known limitation**: `leave_one_author` on `gemini`/`gpt` rows is really equivalent to holding
out an entire synthetic dataset source, not a true test of generalizing away from one author's
writing style, since those datasets have no genuine per-row author. `leave_one_topic` requires
running the trained BERTopic model (`models/saved_topic_model/`) once over the full dataset to
assign `topic_id` before splitting — a Colab-only, GPU-dependent step, not testable locally.

## Future Work

- **Identify the leading account(s) per narrative group** — for each of the 7 narratives, the
  Twitter/Telegram scrapers currently pull from a fixed list of accounts/channels treated
  equally (see `NARRATIVES_ACCOUNTS` in `build_twitter_dataset.py`). A useful extension is to
  determine which account within each group acts as the primary/most influential voice
  ("group leader"), e.g. by engagement volume, retweet/citation frequency by the other accounts
  in the same group, or centrality in a narrative-specific interaction graph:
  - **Zionist**: `Israel`, `IDF`, `StandWithUs`, `AIPAC`, `IsraelMFA`, `AJCGlobal`, `JNS_org`
  - **Resistance**: `khamenei_ir`, `PressTV`, `QudsNen`, `IrnaEnglish`, `TehranTimes79`, `MayadeenEnglish`
  - **Western**: `NATO`, `EU_Commission`, `POTUS`, `StateDept`, `FCDOGovUK`, `GermanyDiplo`
  - **Russian**: `KremlinRussia_E`, `mfa_russia`, `RussiaUN`, `RT_com`, `SputnikInt`, `tassagency_en`
  - **Ukrainian**: `ZelenskyyUa`, `Ukraine`, `DefenceU`, `MFA_Ukraine`, `GeneralStaffUA`, `United24media`
  - **Right-wing**: `FoxNews`, `BenShapiro`, `dailywire`, `Heritage`, `TPUSA`
  - **Left-wing**: `novaramedia`, `BernieSanders`, `jacobin`, `democracynow`, `thenation`
