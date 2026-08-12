import csv
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# Definition of the narratives and the accounts associated with them
NARRATIVES_ACCOUNTS = {
    "Zionist": [
        "Israel",
        "IDF",
        "StandWithUs",
        "AIPAC",
        "IsraelMFA",
        "AJCGlobal",
        "JNS_org",
    ],

    "Resistance": [
        "khamenei_ir",
        "PressTV",
        "QudsNen",
        "IrnaEnglish",
        "TehranTimes79",
        "MayadeenEnglish",
    ],

    "Western": [
        "NATO",
        "EU_Commission",
        "POTUS",
        "StateDept",
        "FCDOGovUK",
        "GermanyDiplo",
    ],

    "Russian": [
        "KremlinRussia_E",
        "mfa_russia",
        "RussiaUN",
        "RT_com",
        "SputnikInt",
        "tassagency_en",
    ],

    "Ukrainian": [
        "ZelenskyyUa",
        "Ukraine",
        "DefenceU",
        "MFA_Ukraine",
        "GeneralStaffUA",
        "United24media",
    ],

    "Right-wing": [
        "FoxNews",
        "BenShapiro",
        "dailywire",
        "Heritage",
        "TPUSA",
    ],

    "Left-wing": [
        "novaramedia",
        "BernieSanders",
        "jacobin",
        "democracynow",
        "thenation",
    ],
}

# Mapping the indices to the narrative names
LABEL_MAP = {
    "Zionist": 0, "Resistance": 1, "Western": 2, "Russian": 3,
    "Ukrainian": 4, "Right-wing": 5, "Left-wing": 6
}

def setup_driver():
    options = webdriver.EdgeOptions()
    options.add_argument("--disable-notifications")
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    return driver


def scrape_twitter_data(target_tweets_per_account=200):
    output_filename = "data/raw/twitter_natural_dataset.csv"

    completed_accounts = set()
    collected_texts = set()

    if os.path.isfile(output_filename):
        try:
            df_existing = pd.read_csv(output_filename)
            collected_texts = set(df_existing['text'].dropna().astype(str).tolist())
            account_counts = df_existing['account'].value_counts().to_dict()
            completed_accounts = {acc for acc, count in account_counts.items() if count >= target_tweets_per_account}
            print(f"Found existing dataset. Skipping completed accounts: {completed_accounts}")
        except Exception as e:
            print("Could not read existing file. Starting fresh.")

    driver = setup_driver()
    file_exists = os.path.isfile(output_filename)

    with open(output_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "label", "narrative_name", "account", "date"])

    try:
        print("Injecting auth token and bypassing login...")

        # Initial navigation to the new domain
        driver.get("https://x.com")
        time.sleep(3)

        # Injecting the cookie with your key - read from an environment variable, never hardcoded.
        _auth_token = os.environ.get("TWITTER_AUTH_TOKEN")
        if not _auth_token:
            raise RuntimeError(
                "Missing TWITTER_AUTH_TOKEN environment variable. Set it with: "
                "$env:TWITTER_AUTH_TOKEN = '<your-auth-token-cookie>' (PowerShell)."
            )
        driver.add_cookie({
            'name': 'auth_token',
            'value': _auth_token,
            'domain': '.x.com'
        })

        # Refresh the page so the login takes effect
        driver.get("https://x.com")
        time.sleep(5)
        print("Authentication successful! Starting data collection...")

        for narrative, accounts in NARRATIVES_ACCOUNTS.items():
            label_idx = LABEL_MAP[narrative]

            for account in accounts:
                if account in completed_accounts:
                    print(f"Skipping @{account} - already reached target.")
                    continue

                print(f"Scraping account: @{account} for narrative: {narrative}...")

                # Navigate to the profile with the new URL
                driver.get(f"https://x.com/{account}")
                time.sleep(30)

                account_tweets_collected = 0
                scroll_attempts = 0
                no_new_tweets_counter = 0

                while account_tweets_collected < target_tweets_per_account and scroll_attempts < 50:
                    previous_collected_count = account_tweets_collected
                    articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                    for article in articles:
                        try:
                            text_elem = article.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                            text = text_elem.text.replace("\n", " ").replace("\r", " ").strip()

                            if text and text not in collected_texts:
                                time_elem = article.find_element(By.TAG_NAME, 'time')
                                tweet_date = time_elem.get_attribute('datetime')

                                collected_texts.add(text)
                                account_tweets_collected += 1

                                with open(output_filename, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([text, label_idx, narrative, account, tweet_date])

                                if account_tweets_collected >= target_tweets_per_account:
                                    break
                        except:
                            continue

                    if account_tweets_collected == previous_collected_count:
                        no_new_tweets_counter += 1
                    else:
                        no_new_tweets_counter = 0

                    if no_new_tweets_counter >= 3:
                        print(
                            f"Reached bottom or blocked at @{account}. Stopping early. Total collected here: {account_tweets_collected}")
                        break

                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(10)
                    scroll_attempts += 1

                print(f"Finished @{account}. Collected {account_tweets_collected} new tweets.")
                time.sleep(10)

    finally:
        driver.quit()
        print(f"\nSuccess! Data collection completed. Dataset saved to {output_filename}")


if __name__ == "__main__":
    scrape_twitter_data(target_tweets_per_account=200)