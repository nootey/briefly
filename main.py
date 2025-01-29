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
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

hug_token = os.getenv("HUGGINGFACE_TOKEN")
if not hug_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment")

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

def get_initial_terms_from_user():

    # TEMP - hard code some values for the meeting used in testing
    return "Lan-Xi, Human Vibration, Bruel & Kjar"

    char_limit = 183  # Approximate character limit for Whisper input
    initial_prompt = []
    print("\nEnter any potential business terms, that might help with transcription")
    print("You can also skip this, just press enter.")

    while True:
        user_input = input("Input terms: ").strip()
        if not user_input:
            break

        initial_prompt = [word.strip() for word in user_input.split(",")]

        if len(initial_prompt) < 1 or len(initial_prompt) > 7:
            print("Info: Please enter between 1 and 7 terms.")
            continue

        total_chars = sum(len(word) for word in initial_prompt)
        if total_chars > char_limit:
            print(f"Error: Total character length ({total_chars}) exceeds the {char_limit}-character limit. Please try again.")
            continue  # Ask for input again

        break

    # Print the resulting array
    print("User has provided the following terms:", initial_prompt)
    return ", ".join(initial_prompt)

def transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt):
    """
    Transcribes audio using Whisper and refines it using GPT-4 spellcheck.
    Returns timestamped transcription.
    """

    print("     --> Transcribing audio...")
    transcription_result = model.transcribe(audio_file_path, prompt=initial_prompt, word_timestamps=True)

    print("     --> Refining transcript with GPT-4 spellcheck...")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription_result["text"]},
        ],
    )

    # return completion.choices[0].message.content
    # Return corrected transcription along with timestamps
    return {"corrected_text": completion.choices[0].message.content, "segments": transcription_result["segments"]}

def transcribe_audio(file, audio_file_path, initial_prompt):

    # audio_file_path = t.prepare_audio_file(file)

    model_name = "large-v3"
    print(f"Loading transcription model: {model_name}")
    model = whisper.load_model(model_name)

    system_prompt = "You are a helpful company assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: " + initial_prompt

    print("Starting transcription: ")
    # result = model.transcribe(audio_file_path, initial_prompt=initial_prompt)
    transcription_result = transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt)

    print("Applying speaker diarization...")
    speaker_segments = diarize_speakers(audio_file_path)

    print("Aligning transcription with speaker segments...")
    aligned_transcription = align_transcription_with_speakers(transcription_result, speaker_segments)

    print("Saving transcription ...")
    # t.save_transcription(transcription_result, file)
    formatted_transcription = "\n".join(f"{seg['speaker']}: {seg['text'].strip()}" for seg in aligned_transcription)
    t.save_transcription(formatted_transcription, file)

    speaker_segments = diarize_speakers(audio_file_path)

    aligned_transcription = align_transcription_with_speakers(transcription_result, speaker_segments)

    print("Saving final transcription ...")
    formatted_transcription = "\n".join(f"{seg['speaker']}: {seg['text'].strip()}" for seg in aligned_transcription)

    final_transcription_file = f"{file}_final_transcription.txt"
    with open(final_transcription_file, "w", encoding="utf-8") as f:
        f.write(formatted_transcription)

    print("Audio transcription complete.")

def diarize_speakers(audio_file):
    """
    Applies speaker diarization using Pyannote and returns speaker segments.
    """

    if not hug_token:
        raise ValueError("Hugging Face token not found. Add it to your .env file.")

    print("     --> Loading pyannote library ...")
    # Load pre-trained diarization pipeline
    pipeline = SpeakerDiarization.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hug_token
    )

    # Run diarization on audio file
    print("     --> Running pyannote library ...")
    diarization = pipeline(audio_file)

    # Store speaker labels and timestamps
    speaker_segments = []
    print("     --> Organizing speaker segments ...")
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "speaker": speaker,
            "start": segment.start,
            "end": segment.end
        })

    return speaker_segments

def align_transcription_with_speakers(transcription_result, speaker_segments):
    """
    Aligns transcribed text with speaker diarization timestamps.
    """

    aligned_transcription = []
    speaker_index = 0
    current_speaker = speaker_segments[speaker_index]

    temp_segment = {"speaker": current_speaker["speaker"], "text": ""}

    for word_data in transcription_result["segments"]:
        start_time = word_data["start"]
        end_time = word_data["end"]
        text = word_data["text"]

        # Move to the correct speaker segment
        while speaker_index < len(speaker_segments) - 1 and start_time >= speaker_segments[speaker_index + 1]["start"]:
            aligned_transcription.append(temp_segment)
            speaker_index += 1
            current_speaker = speaker_segments[speaker_index]
            temp_segment = {"speaker": current_speaker["speaker"], "text": ""}

        # Append text to the current speaker segment
        temp_segment["text"] += " " + text

    # Append the last segment
    aligned_transcription.append(temp_segment)

    return aligned_transcription

def main():

    print_welcome_message()

    print("Checking for required dependencies...")
    run_dependency_tests()
    print("All dependencies are present, proceeding ...")

    initial_prompt = get_initial_terms_from_user()

    file_name = "ims_meeting"
    json_file_path = f"results/transcriptions/{file_name}.json"

    # Prepare the audio file
    audio_file_path = t.prepare_audio_file(file_name)

    print("\nStarting transcription.")
    # Create or load transcript
    if not os.path.exists(json_file_path):
        print(f"Transcript {file_name}.json not found, generating ...")
        transcribe_audio(file_name, audio_file_path, initial_prompt)
    else:
        print(f"Transcript {file_name}.json found, proceeding ...")

    sys.exit()

    with open(json_file_path, "r", encoding="utf-8") as jf:
        transcription_data = json.load(jf)

    transcription = transcription_data.get("transcription", "")

    print(f"Transcription [{file_name}.json] selected")

    # Summarize the transcript
    s.create_transcription_summary(transcription, file_name)

if __name__ == "__main__":
    main()
