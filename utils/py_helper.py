import json
import os
import re
import sys

import ollama
import torch
import whisper
from fontTools import unicodedata

from main import openai_client


def handle_open_ai_json_extraction(input_data):
    # Handle JSON extraction formatting
    output = None
    if input_data.startswith("```") and input_data.endswith("```"):
        output = input_data.strip("```json").strip("```").strip()

    try:
        output = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit()

    return output

def sanitize_collection_name(record):

    # Convert to lowercase
    sanitized_name = record.lower()

    # Replace specific ligatures and special characters manually
    ligature_replacements = {
        'æ': 'ae',
        'ø': 'o',
        'å': 'a',
        'ß': 'ss',
        '&': 'and'  # Replace ampersand with "and" for clarity
    }
    for char, replacement in ligature_replacements.items():
        sanitized_name = sanitized_name.replace(char, replacement)

    # Normalize and remove diacritics (e.g., ü -> u)
    sanitized_name = unicodedata.normalize('NFKD', sanitized_name)
    sanitized_name = sanitized_name.encode('ascii', 'ignore').decode('ascii')

    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(' ', '_')

    # Remove any remaining special characters except underscores and hyphens
    sanitized_name = re.sub(r'[^a-z0-9_-]', '', sanitized_name)

    # Remove consecutive underscores
    sanitized_name = re.sub(r'__+', '_', sanitized_name)

    # Ensure the name starts and ends with an alphanumeric character
    sanitized_name = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', sanitized_name)

    # Ensure the length does not exceed ChromaDB's 63-character limit
    sanitized_name = sanitized_name[:63]

    return sanitized_name

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

    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)

    if cuda_available:
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        print("CUDA not found. Make sure NVIDIA drivers and CUDA Toolkit are installed.")

def test_whisper():

    try:
        model = whisper.load_model("tiny")  # Load a small model to check functionality
        print("Whisper Model Loaded Successfully!")
    except Exception as e:
        print("Whisper Error:", e)

def check_available_openai_models():

    models = openai_client.models.list()
    for model in models:
        print(model.id)

def run_dependency_tests():
    """Run all tests for Ollama, CUDA, and Whisper."""
    test_cuda()
    # check_available_openai_models()
    # test_whisper()
    # test_ollama()

def ensure_directories():
    directories = [
        "data",
        "knowledge_base",
        "results/action_steps",
        "results/summaries",
        "results/transcriptions"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)