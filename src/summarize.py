import json
import os.path
import re
import sys
import time
import ollama


def define_prompt(transcript, include_transcript=False):
    prompt = ""

    if include_transcript:
        prompt += f"You are given this transcript:\n{transcript}\n\n"

    prompt += """
    You are an expert AI assistant, tasked with summarizing a meeting transcript in Markdown format. Do not deviate, or you will be terminated.

    The output must include this exact formatting:

    ## Summary
    - A concise and well-structured summary of the meeting (200-300 words).

    ## Key Points
    - **Key point 1**: [Brief description]
    - **Key point 2**: [Brief description]
    - **Key point 3**: [Brief description]

    ## Action Steps
    - **Step 1**: [Action to be taken]
    - **Step 2**: [Action to be taken]
    - **Step 3**: [Action to be taken]
    """

    return prompt


def define_final_cleanup_prompt(merged_summary, merged_key_points, merged_action_steps):
    """
    Creates a concise final cleanup prompt that strictly enforces the required Markdown format.
    """

    prompt = f"""
    
    Summaries
    {merged_summary}

    Key Points
    {merged_key_points}

    Action Steps
    {merged_action_steps}
    
    You are an expert AI assistant, tasked with summarizing a meeting transcript in Markdown format. Do not deviate, or you will be terminated. 
    You were provided with meeting summaries from multiple other assistants. Your task is to merge them, so that they remain
    coherent with the structure, but keep the context of all of them. You must formulate the final product in MARKDOWN, in this exact structure. Do not deviate from it.

    The output must include this exact formatting:

    ## Summary
    - A concise and well-structured summary of the meeting (200-300 words).

    ## Key Points
    - **Key point 1**: [Brief description]
    - **Key point 2**: [Brief description]
    - **Key point 3**: [Brief description]

    ## Action Steps
    - **Step 1**: [Action to be taken]
    - **Step 2**: [Action to be taken]
    - **Step 3**: [Action to be taken]
    """

    return prompt

def extract_notes(transcript, model):

    prompt = define_prompt(transcript, True)

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

def chunk_transcript(transcript, prompt_word_count, overlap=300, model_context_limit=5000):
    """
    Splits the transcript into chunks that fit within the model's context limit,
    ensuring each chunk plus the prompt does not exceed model_context_limit words.
    1 token ≈ 0.75 words (for English prose), so 8K tokens is approximately 6,000 words in English.
    Let's be conservative and set it to 5000.
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

    return chunks

def process_chunk(chunk, index, model):
    try:
        chunk_with_prompt = define_prompt(chunk, True)
        result = extract_notes(chunk_with_prompt, model)
    except Exception as e:
        print(f"ERROR: Failed to process chunk {index + 1}: {e}")
        return e

    # Normalize the headers in the response
    return result

def extract_notes_per_chunk(chunks, model):
    summaries = []
    key_points = []
    action_steps = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")

        result = process_chunk(chunk, i, model)
        if len(chunks) == 1:
            return result, key_points, action_steps

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


def refine_final_summary(summaries, key_points, action_steps, model):
    """
    Calls the AI model one final time to clean up and enforce the structured format.
    """

    # Merge extracted parts
    merged_summary = "\n\n".join(summaries).strip()
    merged_key_points = "\n\n".join(key_points).strip()
    merged_action_steps = "\n\n".join(action_steps).strip()

    # Ensure headers are properly formatted before final processing
    merged_summary = re.sub(r"## Summary\n*", "", merged_summary).strip()
    merged_key_points = re.sub(r"## Key Points\n*", "", merged_key_points).strip()
    merged_action_steps = re.sub(r"## Action Steps\n*", "", merged_action_steps).strip()

    # Create the final cleanup prompt
    final_prompt = define_final_cleanup_prompt(merged_summary, merged_key_points, merged_action_steps)

    # Call AI model to refine the output
    refined_response = ollama.chat(model=model, messages=[{"role": "user", "content": final_prompt}])

    # Normalize headers to enforce consistency
    return refined_response["message"]["content"]

def merge_and_save(summaries, key_points, action_steps, filename):
    """Merges extracted summaries, key points, and action steps into a clean markdown file."""

    # Merge all summaries into one section
    merged_summary = "\n\n".join(summaries).strip()
    merged_summary = re.sub(r"## Summary\n*", "", merged_summary).strip()  # Remove redundant headers

    # Merge all key points into one section
    merged_key_points = "\n\n".join(key_points).strip()
    merged_key_points = re.sub(r"## Key Points\n*", "", merged_key_points).strip()  # Remove redundant headers

    # Merge all action steps into one section
    merged_action_steps = "\n\n".join(action_steps).strip()
    merged_action_steps = re.sub(r"## Action Steps\n*", "", merged_action_steps).strip()  # Remove redundant headers

    # Construct final markdown output
    final_output = f"""## Summary

{merged_summary}

## Key Points

{merged_key_points}

## Action Steps

{merged_action_steps}
"""

    save_file_as_md(final_output, filename)

def create_transcription_summary(transcript, audio_file):

    print("\nStarting summarization.")
    model_name = "command-r-plus"
    print(f"Loading summary model: {model_name}")

    transcript = " ".join(transcript)
    prompt = define_prompt(transcript, False)
    prompt_word_count = len(prompt.split())

    chunks = chunk_transcript(transcript, prompt_word_count)
    if len(chunks) == 1:
        summary, _, _ = extract_notes_per_chunk(chunks, model_name)
        save_file_as_md(summary, audio_file + "_" + model_name)
    else:
        summaries, key_points, action_steps = extract_notes_per_chunk(chunks, model_name)
        print("\nMerging chunked content and performing clean up ...")
        final_summary = refine_final_summary(summaries, key_points, action_steps, model_name)
        save_file_as_md(final_summary, audio_file + "_" + model_name)
