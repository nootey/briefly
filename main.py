import json
import os
import sys
import ollama
import torch
import whisper
import whisperx
from src import transcribe as t
from src import summarize as s
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

hug_token = os.getenv("HUGGINGFACE_TOKEN")
if not hug_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
# It can be re-enabled by calling
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


    batch_size = 16
    compute_type = "float16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Loading transcription model: {model_name}")
    # model = whisper.load_model(model_name)

    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)

    print("Starting transcription: ")
    # result = model.transcribe(audio_file_path, initial_prompt=initial_prompt)
    # transcription_result = transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt)
    audio = whisperx.load_audio(audio_file_path)
    transcription_result = model.transcribe(audio_file_path, batch_size=batch_size)

    segments = transcription_result.get("segments")

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=transcription_result.get("language"), device=device)
    transcription_result = whisperx.align(transcription_result.get("segments"), model_a, metadata, audio, device, return_char_alignments=False)



    print("Applying speaker diarization...")
    diarization_model = whisperx.DiarizationPipeline(use_auth_token=hug_token, device=device)
    diarize_segments = diarization_model(audio_file_path)
    #
    # print("Aligning transcription with speaker labels...")
    # aligned_transcription = align_transcription_with_speakers(segments, diarization_result)

    result = whisperx.assign_word_speakers(diarize_segments, transcription_result)


    # # Refine transcription using GPT-4
    print("Refining transcript with GPT-4 spellcheck...")
    corrected_transcription = correct_transcription_with_gpt4(result.get("segments"), initial_prompt)

    t.save_diarized_transcription(file, corrected_transcription)

    print("Audio transcription complete.")

def correct_transcription_with_gpt4(aligned_transcription, initial_prompt):
    """
    Uses GPT-4 to refine the transcription while preserving speakers.

    """

    system_prompt = f"You are a transcription assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly, but preserve speaker attribution: {initial_prompt}"
    print(aligned_transcription)
    corrected_transcription = []
    for seg in aligned_transcription:
    #     completion = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         temperature=0,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": seg['text']},
    #         ],
    #     )
    #     corrected_transcription.append({"speaker": seg['speaker'], "text": completion.choices[0].message.content})
          corrected_transcription.append({"speaker": seg.get('speaker'), "text": seg.get('text')})

    return corrected_transcription


def align_transcription_with_speakers(segments, diarization_df):
    """
    Aligns WhisperX transcript with speaker diarization labels.
    If no exact match is found, it assigns the closest speaker.
    """

    aligned_transcription = []
    speaker_map = {}  # Dictionary to map detected speakers to Speaker 1, Speaker 2, etc.
    speaker_counter = 1  # Start naming speakers

    for segment in segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()

        assigned_speaker = "Unknown"

        # Find the speaker with overlapping time
        for _, row in diarization_df.iterrows():
            if (start_time >= row["start"] and start_time <= row["end"]) or (end_time >= row["start"] and end_time <= row["end"]):
                detected_speaker = row["speaker"]

                # Assign a generic "Speaker 1", "Speaker 2" label
                if detected_speaker not in speaker_map:
                    speaker_map[detected_speaker] = f"Speaker {speaker_counter}"
                    speaker_counter += 1

                assigned_speaker = speaker_map[detected_speaker]
                break  # Stop checking once we find the first matching speaker

        aligned_transcription.append({"speaker": assigned_speaker, "text": text})

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
