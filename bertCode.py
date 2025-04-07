#Code written with the help of Kaggle, ChatGPT

import os
import torch
import transformers
import spacy
import csv

nlp = spacy.load("en_core_web_trf") #load English NLP pipeline with transformer-based parser

pronoun_gender = {
    "he": "he/him", "him": "he/him", "his": "he/him",
    "she": "she/her", "her": "she/her", "hers": "she/her",
    "they": "they/them", "them": "they/them", "their": "they/them", "theirs": "they/them"
}

def extract_subject_pronouns(text):
    doc = nlp(text)

    candidates = [token for token in doc if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ == "PROPN"]
    main_subject = candidates[0] if candidates else None

    pronouns = []
    for token in doc:
        if token.pos_ == "PRON" and token.text.lower() in pronoun_gender:
            # Approximate coreference by checking if pronoun is in same sentence
            if main_subject and token.sent == main_subject.sent:
                pronouns.append(token.text.lower())

    # Count and infer gender
    gender_counts = {"he/him": 0, "she/her": 0, "they/them": 0}
    for p in pronouns:
        gender = pronoun_gender[p]
        gender_counts[gender] += 1

    dominant_gender = max(gender_counts, key=gender_counts.get)
    inferred_gender = dominant_gender if gender_counts[dominant_gender] > 0 else "undetermined"

    return {
        "main_subject": main_subject.text if main_subject else None,
        "pronouns": pronouns,
        "inferred_gender": inferred_gender
    }

def process_gender_csv(path, csv_output):
    with open(csv_output, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["filename", "main_subject", "pronouns", "inferred_gender"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                file_path = os.path.join(path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    result = extract_subject_pronouns(text)
                    result["filename"] = filename
                    writer.writerow(result)

path = "limmytalks/transcripts"
csv_output = "genderResults.csv"
process_gender_csv(path, csv_output)