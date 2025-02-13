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

