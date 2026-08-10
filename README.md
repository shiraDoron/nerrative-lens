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
python src/train.py
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
`deep-translator`, `telethon`, `google-genai`.

## Pipeline Overview

1. **Data collection**: `build_twitter_dataset.py` / `build_telegram_dataset.py` scrape
   narrative-labeled posts from a fixed set of accounts/channels per narrative;
   `build_ai_dataset.py` supplements this with synthetic Gemini-generated text.
2. **Preprocessing**: `translate_datasets.py` translates non-English text to English in place.
3. **Training**: `train_topics.py` fits a BERTopic topic model; `train.py` extracts features
   (NER, SRL, emotion/agency, topic/stance, reliability) for every sample, fuses them via
   `fusion.py`'s `NarrativeDetector`, and trains the classifier with early stopping.
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
