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

def save_file_as_md(summary, filename):
    """Saves the given summary text as a markdown file."""
    path = "results/summaries"
    with open(os.path.join(path, filename+".md"), "w", encoding="utf-8") as md_file:
        md_file.write(summary)
    print(f"Summary saved as {filename}.md")
def create_transcription_summary(transcript, audio_file):

    model = "mistral"
    result = extract_notes(transcript, model)

    save_file_as_md(result, audio_file+"_"+model)

