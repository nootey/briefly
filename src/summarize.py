import os.path
import re
import ollama

def define_prompt(transcript, include_transcript=False):
    prompt = ""

    if include_transcript:
        prompt += f"You are given this transcript:\n{transcript}\n\n"

    prompt += """
    You are an expert AI assistant, tasked with summarizing a meeting transcript in Markdown format. Do not deviate, or you will be terminated.

    You carefully provide accurate, factual, thoughtful, nuanced responses, and are brilliant at reasoning. Only output the summary which can include questions asked, any interesting quotes, and any action items that were discussed. 

    Analyze the following meeting transcript. Provide a comprehensive summary paragraph or two (250-400 words) that captures the key points discussed, decisions made, and the overall purpose of the meeting:

    The output must include this exact formatting:

    ## Summary
    - A concise and well-structured summary of the meeting.
    """

    return prompt


def define_final_cleanup_prompt(merged_summary):

    prompt = f"""

    Summaries
    {merged_summary}

    You are an expert AI assistant, tasked with summarizing a meeting transcript in Markdown format. Do not deviate, or you will be terminated. 
    You were provided with meeting summaries from multiple other assistants. Your task is to merge them, so that they remain
    coherent with the structure, but keep the context of all of them. You must formulate the final product in MARKDOWN, in this exact structure. Do not deviate from it.

    The output must include this exact formatting:

    ## Summary
    - A concise and well-structured summary of the meeting.
    """

    return prompt


def extract_notes(transcript, model):
    try:
        # Define the prompt
        prompt = define_prompt(transcript, True)

        # Make the API call to Ollama
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

        # Check if the response is in the expected format
        if "message" not in response or "content" not in response["message"]:
            raise ValueError("Invalid response format from Ollama")

        return response["message"]["content"]

    except KeyError as e:
        print(f"KeyError: Missing expected key in the response: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def chunk_transcript(transcript, prompt_word_count, overlap=300, model_context_limit=5000):

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

    for i, chunk in enumerate(chunks):
        print(f"     --> Processing chunk {i + 1}/{len(chunks)}...")

        result = process_chunk(chunk, i, model)
        if len(chunks) == 1:
            return result

        # Extract sections
        summary_start = result.find("## Summary")

        if summary_start == -1:
            print(f"     --> WARNING: Missing sections in chunk {i + 1}. Skipping.")
            continue

        summaries.append(result[summary_start:].strip())

        print(f"     --> Chunk {i + 1} processed successfully.")

    return summaries


def save_file_as_md(summary, filename):

    path = "results/summaries"
    with open(os.path.join(path, filename + ".md"), "w", encoding="utf-8") as md_file:
        md_file.write(summary)
    print(f"\nSummary saved as {filename}.md")


def refine_final_summary(summaries, model):

    # Merge extracted parts
    merged_summary = "\n\n".join(summaries).strip()

    # Ensure headers are properly formatted before final processing
    merged_summary = re.sub(r"## Summary\n*", "", merged_summary).strip()

    # Create the final cleanup prompt
    final_prompt = define_final_cleanup_prompt(merged_summary)

    # Call AI model to refine the output
    refined_response = ollama.chat(model=model, messages=[{"role": "user", "content": final_prompt}])

    # Normalize headers to enforce consistency
    return refined_response["message"]["content"]


def merge_and_save(summaries, key_points, action_steps, filename):

    # Merge all summaries into one section
    merged_summary = "\n\n".join(summaries).strip()
    merged_summary = re.sub(r"## Summary\n*", "", merged_summary).strip()  # Remove redundant headers

    # Construct final markdown output
    final_output = f"""## Summary
{merged_summary}
"""

    save_file_as_md(final_output, filename)


def create_transcription_summary(transcript, audio_file, model_name):

    print("\nStarting summarization.")
    print(f"Loading summary model: {model_name}")

    transcript = " ".join(transcript)
    prompt = define_prompt(transcript, False)
    prompt_word_count = len(prompt.split())

    print(f"     --> Generating a summary via chunks")
    chunks = chunk_transcript(transcript, prompt_word_count)
    if len(chunks) == 1:
        print("     --> Performing clean up ...")
        final_summary = extract_notes_per_chunk(chunks, model_name)
        # save_file_as_md(summary, audio_file + "_" + model_name)
    else:
        summaries = extract_notes_per_chunk(chunks, model_name)
        print("     --> Merging chunked content and performing clean up ...")
        final_summary = refine_final_summary(summaries, model_name)
        # save_file_as_md(final_summary, audio_file + "_" + model_name)

    return final_summary