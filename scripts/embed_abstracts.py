import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def load_processed_data(path):
    with open(path, "r") as f:
        return json.load(f)

def save_embedded_records(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

def embed_abstracts_specter(data, model_name="allenai/specter"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embedded = []

    for conf_name, papers in data.items():
        for paper in tqdm(papers, desc=f"Embedding {conf_name}"):
            abstract = paper.get("abstract", "").strip()
            if not abstract:
                continue

            inputs = tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)
                embedding = output.last_hidden_state[:, 0, :]  # CLS token
                embedding = embedding.squeeze().cpu().numpy()

            embedded.append({
                "embedding": embedding.tolist(),
                "conference": conf_name,
                "title": paper.get("title", ""),
                "abstract": abstract,
                "venue": paper.get("venue", None),
                "id": paper.get("id", None),
                "keywords": paper.get("keywords", [])
            })

    return embedded

if __name__ == "__main__":
    input_path = "/mnt/data/sara-salamat/generative-topic-evolution/data/processed/cleaned_data_per_conference.json"
    output_path = "/mnt/data/sara-salamat/generative-topic-evolution/data/processed/embedded_records.json"

    print("Loading processed data...")
    data = load_processed_data(input_path)

    print("Embedding abstracts with SPECTER...")
    embedded_records = embed_abstracts_specter(data)

    print(f"Saving {len(embedded_records)} embedded records...")
    save_embedded_records(embedded_records, output_path)

    print("Done.")
