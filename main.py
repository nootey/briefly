import json
import os
import sys

import ollama
import torch
import whisper
from src import transcribe as t
from src import summarize as s


def print_welcome_message():
    message = "Welcome to Briefly!"
    padding = 2
    width = len(message) + padding * 2

    print("#" * (width + 2))
    print("#" + " " * width + "#")
    print("#" + " " * padding + message + " " * padding + "#")
    print("#" + " " * width + "#")
    print("#" * (width + 2))

def test_ollama():
    """Test if Ollama and Phi-3 model are working."""
    try:
        response = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': 'Say Hello, like president Trump would say'}])
        print("Ollama healthcheck:", response['message']['content'])
    except Exception as e:
        print("Ollama Error:", e)

def test_cuda():
    """Check if CUDA is available for PyTorch."""
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)

    if cuda_available:
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        print("CUDA not found. Make sure NVIDIA drivers and CUDA Toolkit are installed.")

def test_whisper():
    """Check if OpenAI's Whisper is installed and working."""
    try:
        model = whisper.load_model("tiny")  # Load a small model to check functionality
        print("Whisper Model Loaded Successfully!")
    except Exception as e:
        print("Whisper Error:", e)

def run_dependency_tests():
    """Run all tests for Ollama, CUDA, and Whisper."""
    # test_ollama()
    test_cuda()
    # test_whisper()


def transcribe_audio(file):
    audio_file_path = t.prepare_audio_file(file)

    # Check if file exists
    if os.path.exists(audio_file_path):
        print(f"Selected file: {audio_file_path} exists, proceeding ...")
    else:
        print(f"File not found: {audio_file_path}")
        sys.exit()

    model_name = "large"
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)

    print("Transcribing audio ...")
    result = model.transcribe(audio_file_path)

    print("Saving transcription ...")
    t.save_transcription(result["text"], file)

    print("Audio transcription complete.")
def main():

    print("\n")
    print_welcome_message()
    print("\n")

    print("Checking for required dependencies...")
    run_dependency_tests()
    print("All dependencies are present, proceeding ...")
    print("\n")

    audio_file = "ims_meeting"
    json_file_path = f"results/transcriptions/{audio_file}.json"

    # Check if transcription already exists
    if not os.path.exists(json_file_path):
        transcribe_audio(audio_file)

    with open(json_file_path, "r", encoding="utf-8") as jf:
        transcription_data = json.load(jf)

    transcription = transcription_data.get("transcription", "")

    print(f"Transcription [{audio_file}] selected")

    s.create_transcription_summary(transcription, audio_file)

if __name__ == "__main__":
    main()
