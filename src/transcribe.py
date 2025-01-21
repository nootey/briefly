import json
import os

def split_into_paragraphs(text, char_limit=300):
    """Splits text into paragraphs based on a character limit,
    ensuring splits occur at sentence boundaries if possible."""
    paragraphs = []
    current_paragraph = ""

    for sentence in text.split(". "):  # Split at sentence boundaries
        if len(current_paragraph) + len(sentence) + 2 <= char_limit:
            current_paragraph += sentence + ". "
        else:
            paragraphs.append(current_paragraph.strip())  # Save paragraph
            current_paragraph = sentence + ". "

    if current_paragraph:  # Add the last paragraph if not empty
        paragraphs.append(current_paragraph.strip())

    return paragraphs
def save_transcription(text, filename):
    """Save transcription as a JSON file using the MP3 filename.

    If a file with the same name exists, it appends an index (e.g., file_1.json).
    """
    json_file_path = f"results/{filename}.json"

    # Ensure unique filename if file already exists
    index = 1
    while os.path.exists(json_file_path):
        json_file_path = f"results/{filename}_{index}.json"
        index += 1

    # Structure JSON data
    paragraphs = split_into_paragraphs(text, char_limit=300)
    data = {"transcription": paragraphs}

    # Save JSON file
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Transcription saved to {json_file_path}")

