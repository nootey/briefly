import json
import os
import sys
import ollama
import torch
import whisper
from tqdm import tqdm

from src import transcribe as t
from src import summarize as s
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def print_welcome_message():
    print("\n")
    message = "Welcome to Briefly!"
    padding = 2
    width = len(message) + padding * 2

    print("#" * (width + 2))
    print("#" + " " * width + "#")
    print("#" + " " * padding + message + " " * padding + "#")
    print("#" + " " * width + "#")
    print("#" * (width + 2))
    print("\n")


def test_ollama():
    """Test if Ollama is running and list available models."""
    try:

        # Get the list of available models
        response = ollama.list()

        # Print the list of models
        print("Ollama is running. Available models:")
        for model in response['models']:
            print(f"- {model}")

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

def check_available_openai_models():
    """
    Displays all available OpenAI models for the given API key.
    """
    models = client.models.list()
    for model in models:
        print(model.id)

def run_dependency_tests():
    """Run all tests for Ollama, CUDA, and Whisper."""
    test_cuda()
    # check_available_openai_models()
    # test_whisper()
    # test_ollama()


# define a wrapper function for seeing how prompts affect transcriptions
def transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt):

    # Step 1: Transcription
    transcribed_text = ""
    print("     --> Transcribing audio...")

    transcription_result = model.transcribe(audio_file_path, prompt=initial_prompt)
    transcribed_text = transcription_result["text"]
    t.save_transcription(transcribed_text, file)

    # Step 2: Spellchecking with GPT-4
    print("     --> Refining transcript ...")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcribed_text},
        ],
    )

    return completion.choices[0].message.content

def transcribe_audio(file, audio_file_path):

    # audio_file_path = t.prepare_audio_file(file)

    model_name = "large-v3"
    print(f"Loading transcription model: {model_name}")
    model = whisper.load_model(model_name)

    # This has a token limit of 244
    initial_prompt = "Lan-Xi, Human Vibration, Bruel & Kjar"
    system_prompt = "You are a helpful company assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: " + initial_prompt

    print("Starting transcription: ")
    # result = model.transcribe(audio_file_path, initial_prompt=initial_prompt)
    result = transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt)

    print("Saving transcription ...")
    # t.save_transcription(result["text"], file)
    t.save_transcription(result, file)

    print("Audio transcription complete.")
def main():

    print_welcome_message()

    print("Checking for required dependencies...")
    run_dependency_tests()
    print("All dependencies are present, proceeding ...")

    file_name = "ims_meeting"
    json_file_path = f"results/transcriptions/{file_name}.json"

    # Prepare the audio file
    audio_file_path = t.prepare_audio_file(file_name)

    print("\nStarting transcription.")
    # Create or load transcript
    if not os.path.exists(json_file_path):
        print(f"Transcript {file_name}.json not found, generating ...")
        transcribe_audio(file_name, audio_file_path)
    else:
        print(f"Transcript {file_name}.json found, proceeding ...")

    with open(json_file_path, "r", encoding="utf-8") as jf:
        transcription_data = json.load(jf)

    transcription = transcription_data.get("transcription", "")

    print(f"Transcription [{file_name}.json] selected")

    # Summarize the transcript
    s.create_transcription_summary(transcription, file_name)

if __name__ == "__main__":
    main()
