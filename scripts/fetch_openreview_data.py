import os
import json
import logging
import openreview

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_notes(client, invitation):
    logging.info(f"Trying invitation: {invitation}")
    try:
        return list(openreview.tools.iterget_notes(client, invitation=invitation))
    except Exception as e:
        logging.error(f"Failed to fetch notes for invitation {invitation}: {e}")
        return []

def fetch_decisions_per_paper(client, submissions, venue):
    """
    For each paper, try to fetch the corresponding decision note.
    """
    decisions = {}
    for note in submissions:
        if not hasattr(note, 'number'):
            continue  # Skip malformed notes
        paper_id = f"{venue}/Paper{note.number}/-/Decision"
        try:
            result = client.get_all_notes(invitation=paper_id)
            if result:
                decisions[note.forum] = result[0].content.get("decision", "")
        except Exception as e:
            logging.warning(f"Could not fetch decision for Paper {note.number}: {e}")
    return decisions

def download_data(data_path, venue, venue_short, client):
    full_path = os.path.join(data_path, venue_short)
    check_dir(full_path)

    # Fetch submissions
    submission_inv = f'{venue}/-/Blind_Submission'
    submissions = get_notes(client, submission_inv)
    logging.info(f"Fetched {len(submissions)} submissions for {venue_short}")
    if not submissions:
        logging.warning(f"No submissions found for {venue_short}")
        return

    # Fetch decisions
    decisions = fetch_decisions_per_paper(client, submissions, venue)
    logging.info(f"Fetched {len(decisions)} decisions for {venue_short}")

    # Merge decision into each submission
    note_dicts = []
    for note in submissions:
        data = note.to_json() if hasattr(note, 'to_json') else note
        data["decision"] = decisions.get(note.forum, None)
        note_dicts.append(data)

    # Save to JSON
    output_path = os.path.join(full_path, f"{venue_short}_notes_with_decisions.json")
    with open(output_path, "w") as f:
        json.dump(note_dicts, f)
    logging.info(f"Saved merged notes with decisions to {output_path}")

if __name__ == '__main__':
    DATA_PATH = '/mnt/data/sara-salamat/generative-topic-evolution/data/raw'

    venues = [
        'NeurIPS.cc/2024/Conference',
        'NeurIPS.cc/2023/Conference',
    ]
    venue_shorts = [
        'neurips2024',
        'neurips2023',
    ]

    client = openreview.Client(baseurl='https://api.openreview.net')

    for venue, short in zip(venues, venue_shorts):
        try:
            download_data(DATA_PATH, venue, short, client)
        except Exception as e:
            logging.error(f"Failed to process {short}: {e}")
