import json
import os
import whisper

from src import transcribe as t
from src import assistant as a
from utils import py_helper as ph
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)


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
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcribed_text},
        ],
    )

    return completion.choices[0].message.content

def transcribe_audio(file, audio_file_path, initial_prompt):

    # audio_file_path = t.prepare_audio_file(file)

    model_name = "large-v3"
    print(f"Loading transcription model: {model_name}")
    model = whisper.load_model(model_name)

    system_prompt = "You are a helpful company assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: " + initial_prompt

    print("Starting transcription: ")
    # result = model.transcribe(audio_file_path, initial_prompt=initial_prompt)
    result = transcribe_with_spellcheck(file, model, audio_file_path, initial_prompt, system_prompt)

    print("Saving transcription ...")
    # t.save_transcription(result["text"], file)
    t.save_transcription(result, file)

    print("Audio transcription complete.")

def get_initial_terms_from_user():

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
def main():

    ph.ensure_directories()
    ph.print_welcome_message()

    print("Checking for required dependencies...")
    ph.run_dependency_tests()
    print("All dependencies are present, proceeding ...")

    initial_prompt = get_initial_terms_from_user()

    file_name = "meeting_example_short"
    json_file_path = f"results/transcriptions/{file_name}.json"
    summary_model_name = "command-r-plus"

    # Prepare the audio file
    audio_file_path = t.prepare_audio_file(file_name)

    print("\nStarting transcription.")
    # Create or load transcript
    if not os.path.exists(json_file_path):
        print(f"Transcript {file_name}.json not found, generating ...")
        transcribe_audio(file_name, audio_file_path, initial_prompt)
    else:
        print(f"Transcript {file_name}.json found, proceeding ...")

    with open(json_file_path, "r", encoding="utf-8") as jf:
        transcription_data = json.load(jf)

    # Summarize the transcript
    transcription = transcription_data.get("transcription", "")
    print(f"Transcription [{file_name}.json] selected")

    print("Generating summary ...")
    summary = a.create_summary(openai_client, "gpt-4o", transcription_data)
    with open(f"results/summaries/{file_name}.md", "w", encoding="utf-8") as md_file:
        md_file.write(summary)

    print("Generating personalized action steps ...")
    action_steps = a.entrypoint(openai_client, transcription_data)

    with open(f"results/action_steps/{file_name}.md", "w", encoding="utf-8") as md_file:
        md_file.write(action_steps)

    print(f"\nSummary saved as {file_name}.md")


if __name__ == "__main__":
    main()
