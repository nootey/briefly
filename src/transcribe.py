import json
import os
import sys

from pydub import AudioSegment

def prepare_audio_file(audio_file):
    directory = "data"
    mp3_dir = os.path.join(directory, audio_file+".mp3")
    if os.path.exists(mp3_dir):
        return mp3_dir

    # Find the file with any extension
    for file in os.listdir(directory):
        if file.startswith(audio_file):
            audio_file_extension = os.path.splitext(file)[1][1:]  # Extract extension without dot
            audio_file_path = os.path.join(directory, file)
            break
    else:
        print("File not found.")
        sys.exit()

    if audio_file_extension == "mp4":
        audio_file = extract_mp3(audio_file_path, mp3_dir)

    return mp3_dir

def extract_mp3(mp4_file, mp3_file):
    """
    Whisper tends to perform better with MP3 files in some cases, compared to WAV, because they are trained on real-world, often compressed, noisy audio data.
    MP3 compression may sometimes remove background noise and artifacts, making speech more distinct.
    """
    if not os.path.exists(mp4_file):
        print(f"Error: {mp4_file} does not exist.")
        return

    print(f"Extracting audio from {mp4_file} to {mp3_file}...")

    # Load the MP4 file
    audio = AudioSegment.from_file(mp4_file, format="mp4")

    # Export with the highest quality MP3 settings
    audio.export(mp3_file, format="mp3", bitrate="320k")

    return mp3_file

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
    json_file_path = f"results/transcriptions/{filename}.json"

    # # Ensure unique filename if file already exists
    # index = 1
    # while os.path.exists(json_file_path):
    #     json_file_path = f"results/transcriptions/{filename}_{index}.json"
    #     index += 1

    # Structure JSON data
    paragraphs = split_into_paragraphs(text, char_limit=300)
    data = {"transcription": paragraphs}

    # Save JSON file
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Transcription saved to {json_file_path}")


def save_diarized_transcription(file, corrected_transcription, char_limit=300):
    """
    Saves the final corrected transcription to a structured JSON format,
    ensuring speaker differentiation and paragraph splitting.
    """

    formatted_transcription = []  # List to store structured transcription data

    for seg in corrected_transcription:
        speaker = seg["speaker"]
        text = seg["text"].strip()

        # Split text into readable paragraphs
        paragraphs = split_into_paragraphs(text, char_limit)

        # Append structured data to list (speaker & paragraphs separately)
        formatted_transcription.append({"speaker": speaker, "text": paragraphs})

    # Prepare JSON file path
    json_file_path = f"results/transcriptions/{file}.json"
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)  # Ensure directory exists

    # Save JSON with structured formatting
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(formatted_transcription, json_file, indent=4, ensure_ascii=False)

    print(f"Transcription successfully saved to {json_file_path}")

