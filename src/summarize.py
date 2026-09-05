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

DETAILED_PROMPT = """\
This is long-form content — a podcast, interview, lecture, or similar — covering \
multiple topics, so summarize it thoroughly instead of compressing everything into a \
short overview.

Write 150-300 words of markdown under a `## Overview` heading. Cover the format, the \
speakers, and the overall arc. Stay factual — do not invent details that are not in the \
transcript.

Then scan the transcript for the distinct topics or segments it moves through. Under a \
`## Topics` heading, add one `### <topic title>` subheading per topic, in the order they \
come up, each followed by a paragraph of 3-6 sentences covering what was actually said. \
Use as many topics as the transcript supports — do not force a fixed count.

Then, under a `## Key Concepts` heading, list up to 5 of the ideas the transcript leans \
on most, one per line as `- **Term** — one sentence on what it means here`. Fewer than \
five is fine when the transcript does not support five.

Output only those three sections.

Transcript:
{transcript}
"""

CHUNK_DETAILED_PROMPT = """\
The transcript below is one part of a longer piece of long-form content — a podcast, \
interview, lecture, or similar covering multiple topics.

Write 100-200 words of markdown under a `## Overview` heading summarizing this part. \
Stay factual — do not invent details that are not in the transcript.

Then scan this part for the distinct topics it covers. Under a `## Topics` heading, add \
one `### <topic title>` subheading per topic, each followed by a paragraph of 2-4 \
sentences covering what was actually said. Use as many topics as this part supports.

Output only those two sections.

Transcript part:
{transcript}
"""

OVERVIEW_MERGE_PROMPT = """\
The partial overviews below each describe one consecutive part of one long, long-form \
transcript (a podcast, interview, lecture, or similar).

Merge them into a single coherent `## Overview` covering the whole recording — the \
format, the speakers, and the overall arc — dropping repetition while keeping every \
distinct point made across the parts.

Output only that section.

Partial overviews:
{transcript}
"""

TOPICS_MERGE_PROMPT = """\
The notes below cover consecutive parts of one long, long-form transcript. Each part's \
notes include a `## Topics` section listing the distinct topics discussed in that part, \
with a paragraph on each.

Consolidate all of them into a single `## Topics` heading. Add one `### <topic title>` \
subheading per distinct topic, ordered by where it first comes up, each followed by a \
paragraph that combines everything said about it across every part where it appears. \
Merge topics that are the same idea even if the parts titled them differently; keep \
topics that are genuinely distinct separate, even if they only appear in one part.

Output only that section.

Notes on each part:
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

    if config.prompt:
        template, detailed = config.prompt, False
        if "{transcript}" not in template:
            raise ValueError("summarize.prompt must contain the {transcript} placeholder")
    else:
        detailed = len(transcript) >= config.topics_min_chars
        template = DETAILED_PROMPT if detailed else SUMMARY_PROMPT

    if len(chunks) == 1:
        return _complete(template.format(transcript=chunks[0]), config, api_key)

    print(f"  transcript split into {len(chunks)} chunks")
    chunk_template = CHUNK_DETAILED_PROMPT if detailed else template
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  summarizing chunk {i}/{len(chunks)}")
        partials.append(_complete(chunk_template.format(transcript=chunk), config, api_key))

    notes = "\n\n".join(partials)

    if detailed:
        print("  merging overviews")
        overview = _complete(OVERVIEW_MERGE_PROMPT.format(transcript=notes), config, api_key)
        print("  consolidating topics")
        topics = _complete(TOPICS_MERGE_PROMPT.format(transcript=notes), config, api_key)
        print("  extracting key concepts")
        concepts = _complete(
            CONCEPTS_PROMPT.format(summary=overview, transcript=notes), config, api_key
        )
        return f"{overview}\n\n{topics}\n\n{concepts}"

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
