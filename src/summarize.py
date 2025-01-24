import json
import os.path
import sys

import ollama

def extract_notes(transcript, model):
    prompt = f"""
    You are an expert assistant summarizing a meeting transcript. You carefully provide accurate, factual, thoughtful, nuanced responses, and are brilliant at reasoning.
    
    Extract the following structured information:

    1. **Summary**: A comprehensive summary (250-400 words) covering key points discussed, decisions made, and the overall purpose of the meeting.
    2. **Key Discussion Points**: List the major topics covered, including important decisions, arguments, or insights.
    3. **Action Steps**: Clearly outline tasks that need to be completed, specifying responsible parties if mentioned.

    **Format your response in Markdown as follows:**
    
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

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

def chunk_transcript(transcript, max_length=1500, overlap=200):
    """Splits the transcript into chunks of max_length words, with overlap."""
    words = transcript.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_length, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_length - overlap  # Shift start position to include overlap

    return chunks

def save_file_as_md(summary, filename):
    """Saves the given summary text as a markdown file."""
    path = "results/summaries"
    with open(os.path.join(path, filename+".md"), "w", encoding="utf-8") as md_file:
        md_file.write(summary)
    print(f"Summary saved as {filename}.md")

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

def extract_notes_per_chunk(chunks, model):
    summaries = []
    key_points = []
    action_steps = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        result = extract_notes(chunk, model)

        # Extract sections
        summary_start = result.find("## Summary")
        key_points_start = result.find("## Key Points")
        action_steps_start = result.find("## Action Steps")

        if summary_start != -1 and key_points_start != -1 and action_steps_start != -1:
            summaries.append(result[summary_start:key_points_start].strip())
            key_points.append(result[key_points_start:action_steps_start].strip())
            action_steps.append(result[action_steps_start:].strip())

    return summaries, key_points, action_steps

def create_transcription_summary(transcript, audio_file):

    model = "command-r-plus"
    # result = extract_notes(transcript, model)
    # save_file_as_md(result, audio_file+"_"+model)

    transcript = " ".join(transcript)
    chunks = chunk_transcript(transcript)
    summaries, key_points, action_steps = extract_notes_per_chunk(chunks, model)
    merge_and_save(summaries, key_points, action_steps, audio_file + "_" + model)



