from src.config import SummarizeConfig

SUMMARY_PROMPT = """\
Summarize the transcript below.

Write 250-500 words of markdown under a single `## Summary` heading. Stay factual — do \
not invent details that are not in the transcript.

Then, under a `## Key Concepts` heading, list up to 5 of the ideas the transcript leans \
on most, one per line as `- **Term** — one sentence on what it means here`. Fewer than \
five is fine when the transcript does not support five.

Output only those two sections.

Transcript:
{transcript}
"""

MERGE_PROMPT = """\
The partial summaries below describe consecutive parts of one long transcript.

Merge them into a single coherent summary under one `## Summary` heading, dropping \
repetition while keeping every distinct point. Ignore their `## Key Concepts` sections; \
those are handled separately.

Output only that section.

Partial summaries:
{transcript}
"""

CONCEPTS_PROMPT = """\
Below is a summary of one long transcript, followed by the notes taken on each of its \
parts.

From both, produce a single `## Key Concepts` list of up to 5 entries, one per line as \
`- **Term** — one sentence on what it means here`. Prefer ideas that run through the \
whole transcript over ones confined to a single part, and collapse entries that name \
the same idea. Take the wording of each term from the notes, which sit closer to what \
was actually said. Fewer than five is fine.

Output only that section.

Summary:
{summary}

Notes on each part:
{transcript}
"""


def run(transcript: str, config: SummarizeConfig, api_key: str | None = None) -> str:
    chunks = _chunk(transcript, config.chunk_chars)
    template = config.prompt or SUMMARY_PROMPT
    if "{transcript}" not in template:
        raise ValueError("summarize.prompt must contain the {transcript} placeholder")

    if len(chunks) == 1:
        return _complete(template.format(transcript=chunks[0]), config, api_key)

    print(f"  transcript split into {len(chunks)} chunks")
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  summarizing chunk {i}/{len(chunks)}")
        partials.append(_complete(template.format(transcript=chunk), config, api_key))

    notes = "\n\n".join(partials)
    print("  merging partial summaries")
    summary = _complete(MERGE_PROMPT.format(transcript=notes), config, api_key)

    print("  extracting key concepts")
    concepts = _complete(
        CONCEPTS_PROMPT.format(summary=summary, transcript=notes), config, api_key
    )
    return f"{summary}\n\n{concepts}"


def _chunk(text: str, limit: int) -> list[str]:
    """Split on paragraph, then sentence, then hard boundaries under `limit`."""
    if limit <= 0:
        raise ValueError("summarize.chunk_chars must be positive")
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for piece in _pieces(text, limit):
        if current and len(current) + len(piece) > limit:
            chunks.append(current.strip())
            current = ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _pieces(text: str, limit: int):
    """Yield fragments no longer than `limit`, preferring natural break points."""
    for paragraph in text.splitlines(keepends=True):
        if len(paragraph) <= limit:
            yield paragraph
            continue
        for sentence in paragraph.replace(". ", ".\x00").split("\x00"):
            if len(sentence) <= limit:
                yield sentence
                continue
            # No natural boundary left (e.g. an unpunctuated wall of text).
            for i in range(0, len(sentence), limit):
                yield sentence[i : i + limit]


def _complete(prompt: str, config: SummarizeConfig, api_key: str | None) -> str:
    from openai import APIStatusError, OpenAI

    client = OpenAI(
        base_url=config.base_url,
        # Local endpoints (Ollama, vLLM, llama.cpp) ignore the key but the SDK still insists on a non-empty one.
        api_key=api_key or "no-key-required",
        max_retries=8,
    )
    try:
        response = client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIStatusError as exc:
        if exc.status_code != 413:
            raise
        # Not retryable: one request simply exceeded what the tier accepts.
        raise RuntimeError(
            f"{config.model} rejected a {len(prompt):,}-character request as too "
            f"large. Lower summarize.chunk_chars in config.yaml — the provider's "
            f"reply says what it will accept: {exc}"
        ) from exc
    return response.choices[0].message.content.strip()
