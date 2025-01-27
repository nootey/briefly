import json
import os.path
import re
import sys
import time
import ollama


def define_prompt(transcript):

    prompt = f"""
    You are an expert assistant summarizing a meeting transcript. You carefully provide accurate, factual, thoughtful, nuanced responses.

    You must return your response in strict Markdown format. DO NOT DEVIATE.
    Make sure to include:
    - "## Summary"
    - "## Key Points"
    - "## Action Steps"
    Failure to follow this format will result in rejection.
    
    Extract the following structured information:

    1. **Summary**: A comprehensive summary (250-450 words) covering key points discussed, decisions made, and the overall purpose of the meeting.
    2. **Key Discussion Points**: List the major topics covered, including important decisions, arguments, or insights.
    3. **Action Steps**: Clearly outline tasks that need to be completed, specifying responsible parties if mentioned.

    **Format your response in Markdown as follows! Your response MUST include these headers:**

    ```
    ## Summary

    [Provide a well-structured summary]

    ## Key Points

    - **Key point 1**: [Brief description]
    - **Key point 2**: [Brief description]
    - **Key point 3**: [Brief description]

    ## Action Steps

    - **Step 1**: [Action to be taken]
    - **Step 2**: [Action to be taken]
    - **Step 3**: [Action to be taken]
    ```

    **Transcript:**
    {transcript}
    """

    return prompt

def extract_notes(transcript, model):

    prompt = define_prompt(transcript)

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def chunk_transcript(transcript, prompt_word_count, overlap=400, model_context_limit=3500):
    """
    Splits the transcript into chunks that fit within the model's context limit,
    ensuring each chunk plus the prompt does not exceed model_context_limit words.
    """

    # Compute the maximum chunk size allowed
    max_chunk_size = model_context_limit - prompt_word_count
    if max_chunk_size <= overlap:
        raise ValueError("Adjusted max_chunk_size must be greater than overlap")

    words = transcript.split()  # Split transcript into words
    chunks = []
    start = 0
    index = 1
    total_words = len(words)

    print(f"\nTranscript character count: {len(words)}")
    print(f"Prompt character count: {prompt_word_count}")
    print(f"Calculated max chunk size: {max_chunk_size}")

    while start < total_words:
        end = min(start + max_chunk_size, total_words)  # Ensure we don’t exceed the total word count

        print(f"Chunk {index} starts at character: {start}, and ends at character: {end}")

        chunk = " ".join(words[start:end])

        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
            print(f"Appended chunk {index} with character count: {len(chunk.split())}")
        else:
            print(f"Skipping empty chunk {index}")

        start += max_chunk_size - overlap  # Move forward while keeping overlap
        index += 1

    print("\n")
    return chunks

def extract_notes_per_chunk(chunks, model):
    summaries = []
    key_points = []
    action_steps = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        try:
            chunk_with_prompt = define_prompt(chunk)
            result = extract_notes(chunk_with_prompt, model)
        except Exception as e:
            print(f"ERROR: Failed to process chunk {i + 1}: {e}")
            continue

        if not result or "## Summary" not in result:
            print(f"WARNING: Chunk {i + 1} contains invalid response.")
            continue

        # Extract sections
        summary_start = result.find("## Summary")
        key_points_start = result.find("## Key Points")
        action_steps_start = result.find("## Action Steps")

        if summary_start == -1 or key_points_start == -1 or action_steps_start == -1:
            print(f"WARNING: Missing sections in chunk {i + 1}. Skipping.")
            continue

        summaries.append(result[summary_start:key_points_start].strip())
        key_points.append(result[key_points_start:action_steps_start].strip())
        action_steps.append(result[action_steps_start:].strip())

        print(f"Chunk {i + 1} processed successfully.")

    return summaries, key_points, action_steps


def save_file_as_md(summary, filename):
    """Saves the given summary text as a markdown file."""
    path = "results/summaries"
    with open(os.path.join(path, filename + ".md"), "w", encoding="utf-8") as md_file:
        md_file.write(summary)
    print(f"\nSummary saved as {filename}.md")


def merge_and_save(summaries, key_points, action_steps, filename):
    """Merges extracted summaries, key points, and action steps into a final markdown file."""
    final_summary = "\n\n".join(summaries)
    final_key_points = "\n\n".join(key_points)
    final_action_steps = "\n\n".join(action_steps)

    final_output = f"""
{final_summary}

{final_key_points}

{final_action_steps}
"""

    save_file_as_md(final_output, filename)

def create_transcription_summary(transcript, audio_file):

    model_name = "qwen2"
    print(f"Loading summary model: {model_name}")

    transcript = " ".join(transcript)
    prompt = define_prompt(transcript)
    prompt_word_count = len(prompt.split())

    chunks = chunk_transcript(transcript, prompt_word_count)
    summaries, key_points, action_steps = extract_notes_per_chunk(chunks, model_name)
    merge_and_save(summaries, key_points, action_steps, audio_file + "_" + model_name)



