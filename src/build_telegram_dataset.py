import csv
import os
from telethon.sync import TelegramClient
from telethon.errors import ChannelPrivateError, UsernameInvalidError

from build_twitter_dataset import NARRATIVES_ACCOUNTS
from config import NARRATIVES

# Personal access credentials you obtained from the site - read from environment variables, never hardcoded.
_API_ID = os.environ.get("TELEGRAM_API_ID")
_API_HASH = os.environ.get("TELEGRAM_API_HASH")
if not _API_ID or not _API_HASH:
    raise RuntimeError(
        "Missing TELEGRAM_API_ID / TELEGRAM_API_HASH environment variables. Set them with: "
        "$env:TELEGRAM_API_ID = '<id>'; $env:TELEGRAM_API_HASH = '<hash>' (PowerShell)."
    )
API_ID = int(_API_ID)
API_HASH = _API_HASH

NARRATIVES_CHANNELS = {
    "Zionist": ["southfirstresponders", "TheJerusalemPost", "BringThemHomeNow",
                "abualiexpress", "HananyaNaftali", "OpenSourceIntel"],
    # Resistance = the regional "Axis of Resistance". NOTE: AlJazeeraEnglish is borderline per
    # the labeling methodology - it provides a lot of Palestinian/critical content but isn't as
    # clean a "Resistance" source as PressTV/Al Mayadeen/QudsNen, and may add noise to the
    # ground truth. Left in for now per the user's decision, with this note for attention
    # during analysis/labeling.
    # TODO: missing sources from Yemen (Al Masirah English) and additional ones from Iraq/Lebanon
    # (Al-Manar English etc.) - an exact handle is needed before adding (don't guess usernames).
    "Resistance": ["ResistanceNewsNetwork", "AlJazeeraEnglish", "TasnimNewsEN",
                   "AlMayadeenEnglish", "PressTV", "QudsNen", "Palestine_Chronicle"],
    "Western": ["WashingtonPost", "Bloomberg", "spectatorindex", "euronews_eng"],
    "Russian": ["Rybar", "IntelSlavaZ", "DDGeopolitics", "Slavyangrad", "mod_russia_en",
                "MariaZakharova", "Readovka", "geopolitics_live"],
    "Ukrainian": ["UkraineNow", "SuspilneNews", "United24Media", "Babel",
                  "UkraineWorld", "ukraine_inc", "KyivIndependent"],
    "Right-wing": ["BreitbartNews", "TheEpochTimes", "Newsmax", "OneAmericaNews",
                   "CharlieKirk", "ThePostMillennial", "WesternJournal", "@KoheletForum"],
    "Left-wing": ["TheIntercept", "ViceNews", "DemocracyDocket", "@MiddleEastEye_TG"]
}


LABEL_MAP = {
    "Zionist": 0,"Resistance": 1,  "Western": 2, "Russian": 3,
    "Ukrainian": 4, "Right-wing": 5, "Left-wing": 6
}


def scrape_telegram_data(target_messages_per_channel=200):
    output_filename = "data/raw/telegram_natural_dataset.csv"

    file_exists = os.path.isfile(output_filename)
    with open(output_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "label", "narrative_name", "account", "date"])

    print("Connecting to Telegram...")

    with TelegramClient('narrative_research_session', API_ID, API_HASH) as client:
        print("Connected successfully!\n")

        for narrative, channels in NARRATIVES_CHANNELS.items():
            label_idx = LABEL_MAP[narrative]

            for channel in channels:
                print(f"Scraping channel: @{channel} for narrative: {narrative}...")
                collected_count = 0

                try:
                    for message in client.iter_messages(channel, limit=target_messages_per_channel * 2):
                        if collected_count >= target_messages_per_channel:
                            break

                        if message.message and len(message.message.strip()) > 10:
                            text = message.message.replace("\n", " ").replace("\r", " ").strip()
                            msg_date = message.date.isoformat()

                            with open(output_filename, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([text, label_idx, narrative, channel, msg_date])

                            collected_count += 1

                    print(f"Finished @{channel}. Collected {collected_count} messages.")

                except (ValueError, ChannelPrivateError, UsernameInvalidError) as e:
                    print(f"Skipping @{channel} - Channel not found, private, or invalid handle.")
                except Exception as e:
                    print(f"An unexpected error occurred with @{channel}: {e}")

    print(f"\nData collection completed. Dataset saved to {output_filename}")


if __name__ == "__main__":
    scrape_telegram_data(target_messages_per_channel=200)