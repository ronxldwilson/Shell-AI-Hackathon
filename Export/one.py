import requests
import time
import os
from bs4 import BeautifulSoup

# --- ⚙️ CONFIGURATION - YOU MUST CHANGE THIS ---
# Paste the entire 'cookie' string from your browser's developer tools here.
COOKIE_STRING = """HE_UTS_ID_CL=907bc1d0b55f4ebab8b2cd98484e7ad89e9959712af04b938c74f91ea6206935; ulang="ZW4tdXM="; plang="ZW4tdXM="; _gcl_au=1.1.1134164221.1751717249; hubspotutk=96197a8173ab7da71e73cc5195cb7408; HE_USER_EXISTS=True; challengesUTM="event=shellai-hackathon-2025&"; _fbp=fb.1.1751717306244.682955746548287378; _clck=1ceirzz%7C2%7Cfxe%7C0%7C2012; __hstc=226601858.96197a8173ab7da71e73cc5195cb7408.1751717255872.1751717255872.1751885182560.2; csrfToken=zCai2eBhNkJQj0thMVSWCUtiDxM7XZ77xIF9loPQrdB9s5xVyfWy6ANBGtHxMIJx; lordoftherings=c878a716007b5f8d87e38f80d6d41e01:ntt0dpkrokzt2gmn5mj56ylpga8s5cnd; HE_IDENTIFIER=12036148; _vwo_uuid_v2=D703D7B1762685483245A0B414228C1C1|92907e74be79ec18bb5c7628f0e0cee4; fs-consent-ad_storage=true; fs-consent-ad_user_data=true; fs-consent-ad_personalization=true; fs-consent-analytics_storage=true; fs-consent-functionality_storage=true; fs-consent-personalization_storage=true; fs-consent-security_storage=true; cb-enabled=accepted; ins_seen_2473303_12036148=1; _rdt_uuid=1751717250590.0248c818-6667-4a37-9ca6-aed3747495b5; _uetvid=a1601300599811f0b42d1726581b2648; COMMUNITY_ONBOARDING_DONE_12036148=True; HE_UTS_ID_LP="/challenges/competitive/shellai-hackathon-2025/submissions/"; HE_UTS_ID=370da77c18514a9fb65b0c7ed9616fb40e86b05ede8a4e3baddbd00ff62dd2a5; user_tz=Asia/Damascus; _gid=GA1.2.771115434.1752802854; _gat=1; _ga=GA1.1.222438919.1751717247; _ga_18S5HG42RB=GS2.1.s1752802858$o13$g1$t1752804201$j59$l0$h1414617944; _ga_58PWML00M0=GS2.2.s1752802857$o7$g1$t1752804202$j59$l0$h0; _ga_R0HG95ZQ2S=GS2.2.s1752802857$o7$g1$t1752804202$j59$l0$h0; mp_4c30b8635363027dfb780d5a61785112_mixpanel=%7B%22distinct_id%22%3A%22%24device%3A1eda195a-3e7c-4f1b-bc3b-1445955be731%22%2C%22%24device_id%22%3A%221eda195a-3e7c-4f1b-bc3b-1445955be731%22%2C%22url_path%22%3A%22%2Fchallenges%2Fcompetitive%2Fshellai-hackathon-2025%2Fmachine-learning%2Ffuel-blend-properties-prediction-challenge-8-c15d7a46%2F%22%2C%22%24initial_referrer%22%3A%22%24direct%22%2C%22%24initial_referring_domain%22%3A%22%24direct%22%2C%22__mps%22%3A%7B%7D%2C%22__mpso%22%3A%7B%22%24initial_referrer%22%3A%22%24direct%22%2C%22%24initial_referring_domain%22%3A%22%24direct%22%7D%2C%22__mpus%22%3A%7B%7D%2C%22__mpa%22%3A%7B%7D%2C%22__mpu%22%3A%7B%7D%2C%22__mpr%22%3A%5B%5D%2C%22__mpap%22%3A%5B%5D%2C%22%24search_engine%22%3A%22google%22%2C%22username%22%3A%22naeem.a%22%2C%22email%22%3A%22naeem.a%40laksor.vip%22%2C%22date_of_joining%22%3A%22Jul%2004%2C%202025%2010%3A36%20AM%22%2C%22user_bucket%22%3A%22120%22%7D; piratesofthecaribbean=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im5hZWVtLmFAbGFrc29yLnZpcCIsInVzZXJfaWQiOjEyMDM2MTQ4LCJleHAiOjE3NTM0MDkwMTEsImlzcyI6ImhlLXNwcmludCJ9.2c8wGujjhl3n_dN9crdUeGEysE4gZ4yqYSkisFJoSLs"""
# -------------------------------------------------

# URLs from your request info
CHALLENGE_URL = "https://www.hackerearth.com/challenges/competitive/shellai-hackathon-2025/machine-learning/fuel-blend-properties-prediction-challenge-8-c15d7a46/"
SUBMIT_URL = "https://www.hackerearth.com/submit-ml/AJAX/29134/"
BASE_URL = "https://www.hackerearth.com"




def submit_and_get_score(csv_file_path: str, session: requests.Session) -> str:
    """
    Submits a CSV file and polls for the score.
    """
    if not os.path.exists(csv_file_path):
        return f"❌ Error: File not found at '{csv_file_path}'"

    try:
        print(f"\n➡️  Submitting file: {csv_file_path}")

        # Get challenge page to update CSRF token
        response_get = session.get(CHALLENGE_URL)
        response_get.raise_for_status()
        soup = BeautifulSoup(response_get.text, 'html.parser')

        csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
        session.headers.update({"X-CSRFToken": csrf_token})

        # Prepare submission
        form_data = {"csrfmiddlewaretoken": csrf_token, "submit": "Submit & Evaluate"}
        with open(csv_file_path, 'rb') as f:
            files = {'upload': (os.path.basename(csv_file_path), f, 'text/csv')}
            response_post = session.post(SUBMIT_URL, data=form_data, files=files)
            response_post.raise_for_status()

        # Extract polling URL
        post_json = response_post.json()
        submission_html = post_json.get("data", "")
        soup = BeautifulSoup(submission_html, 'html.parser')

        poll_url_div = soup.find('div', {'id': 'submission-result-poll-url'})
        if not poll_url_div or not poll_url_div.get('url'):
            return "❌ Error: Could not find the result URL."

        result_url = BASE_URL + poll_url_div['url']
        print(f"✅ Submission accepted. Polling: {result_url}")

        # Poll for result
        max_polls = 60
        for i in range(max_polls):
            poll_response = session.get(result_url)
            poll_response.raise_for_status()
            result_html = poll_response.text

            if "Evaluating..." not in result_html:
                final_soup = BeautifulSoup(result_html, 'html.parser')
                score_heading = final_soup.find('div', class_='content-heading-bold', string='Score')
                if score_heading and score_heading.find_next_sibling('div'):
                    score = score_heading.find_next_sibling('div').text.strip()
                    return f"🏆 Score for '{os.path.basename(csv_file_path)}': {score}"
                else:
                    return "❌ Error: Could not parse the final score."

            print(f"   Poll {i+1}/60: Still evaluating...")
            time.sleep(5)

        return "❌ Error: Polling timed out."

    except Exception as e:
        return f"❌ Exception: {e}"


if __name__ == "__main__":
    # === Add your file names here ===
    submission_files = [
        "submission_319_20250720_091959.csv",
        "submission_125_20250720_052805.csv",
        "submission_425_20250720_111937.csv",
        "submission_24_20250720_033833.csv",
        "submission_400_20250720_105031.csv",
        "submission_89_20250720_044150.csv",
        "submission_393_20250720_104214.csv",
        "submission_330_20250720_093224.csv",
        "submission_199_20250720_065301.csv",
        "submission_361_20250720_100532.csv",
        "submission_54_20250720_042025.csv",
        "submission_162_20250720_060958.csv",
        "submission_424_20250720_111922.csv",
        "submission_394_20250720_104227.csv",
        "submission_337_20250720_094053.csv",
        "submission_323_20250720_092408.csv",
        "submission_53_20250720_042011.csv",
        "submission_356_20250720_100116.csv",
        "submission_344_20250720_094859.csv",
        "submission_404_20250720_105452.csv",
        "submission_69_20250720_043102.csv",
        "submission_410_20250720_110257.csv",
        "submission_244_20250720_074929.csv"

    ]

    # === Basic validation ===
    if "YOUR_FULL_COOKIE_STRING_HERE" in COOKIE_STRING:
        print("❌ Please set COOKIE_STRING.")
        exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Cookie": COOKIE_STRING,
        "Origin": BASE_URL,
        "Referer": CHALLENGE_URL,
        "X-Requested-With": "XMLHttpRequest"
    })

    for idx, file_path in enumerate(submission_files):
        result = submit_and_get_score(file_path, session)
        print("=" * 40)
        print(result)
        print("=" * 40)

        # If not last file, wait 60s before next
        if idx < len(submission_files) - 1:
            print("⏳ Waiting 60 seconds before next submission...")
            time.sleep(15)