import openreview
import os
import json
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("openreview_data_downloader.log"),
        logging.StreamHandler()
    ]
)

# OpenReview API v2 client
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username='',
    password=''
)

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_data(venue_id, retries=5):
    """
    Fetch all submissions for a venue using its submission invitation.
    """
    for attempt in range(retries):
        try:
            logging.info(f"Fetching data for venue: {venue_id} (Attempt {attempt + 1}/{retries})")
            venue_group = client.get_group(venue_id)
            submission_name = venue_group.content['submission_name']['value']
            submissions = client.get_all_notes(
                invitation=f'{venue_id}/-/{submission_name}',
                details=None
            )
            return [note.to_json() for note in submissions]
        except openreview.OpenReviewException as e:
            if "RateLimitError" in str(e):
                logging.warning("Rate limit hit. Retrying in 30 seconds...")
                time.sleep(30)
            else:
                logging.error(f"OpenReviewException: {e}")
                raise
        except Exception as e:
            logging.error(f"Error fetching data for venue {venue_id}: {e}")
            raise
    logging.error(f"Failed to fetch data for {venue_id} after {retries} attempts.")
    return []

if __name__ == '__main__':
    venues = [
        'NeurIPS.cc/2024/Conference',
        'NeurIPS.cc/2023/Conference'
    ]
    venue_shorts = [
        'neurips2024',
        'neurips2023'
    ]

    for venue_id, venue_short in zip(venues, venue_shorts):
        try:
            full_path = os.path.join('/mnt/data/sara-salamat/generative-topic-evolution/data/raw', venue_short)
            check_dir(full_path)
            submissions = get_data(venue_id)

            output_file = os.path.join(full_path, f'{venue_short}_notes_with_decisions.json')
            with open(output_file, 'w') as f:
                json.dump(submissions, f)

            logging.info(f"Saved {len(submissions)} submissions to {output_file}")
        except Exception as e:
            logging.error(f"Error processing {venue_short}: {e}")
