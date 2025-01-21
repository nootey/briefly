import os
import sys

import whisper
from src import transcribe as t


def print_welcome_message():
    message = "Welcome to Briefly!"
    padding = 2
    width = len(message) + padding * 2

    print("#" * (width + 2))
    print("#" + " " * width + "#")
    print("#" + " " * padding + message + " " * padding + "#")
    print("#" + " " * width + "#")
    print("#" * (width + 2))

def main():

    print_welcome_message()

    audio_file = "meeting_example_short"
    audio_file_extension = "mp3"
    audio_file_path = f"data/{audio_file}.{audio_file_extension}"

    # Check if file exists
    if os.path.exists(audio_file_path):
        print(f"Selected file: {audio_file_path} exists, proceeding ...")
    else:
        print(f"File not found: {audio_file_path}")
        sys.exit()

    model_name = "small"
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)

    print("Transcribing audio ...")
    result = model.transcribe(audio_file_path)

    print("Saving transcription ...")
    t.save_transcription(result["text"], audio_file)

    print("Audio transcription complete.")

if __name__ == "__main__":
    main()
