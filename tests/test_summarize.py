import sys
from types import ModuleType, SimpleNamespace

import pytest

from src import summarize
from src.config import SummarizeConfig


@pytest.fixture
def config():
    return SummarizeConfig(chunk_chars=100)


@pytest.fixture
def prompts(monkeypatch):
    """Record every prompt sent, and answer with a marker instead of calling out."""
    seen = []

    def fake_complete(prompt, config, api_key):
        seen.append(prompt)
        return f"reply {len(seen)}"

    monkeypatch.setattr(summarize, "_complete", fake_complete)
    return seen


def test_short_transcript_is_one_call(config, prompts):
    result = summarize.run("a short transcript", config, "key")

    assert result == "reply 1"
    assert len(prompts) == 1
    assert "a short transcript" in prompts[0]


def test_long_transcript_is_chunked_merged_and_distilled(config, prompts):
    transcript = "\n".join(f"paragraph {i} " * 3 for i in range(40))

    result = summarize.run(transcript, config, "key")

    # n chunk calls, then the merge, then the concepts pass.
    assert len(prompts) > 3
    assert summarize.MERGE_PROMPT[:40] in prompts[-2]
    assert summarize.CONCEPTS_PROMPT[:40] in prompts[-1]
    merge, concepts = f"reply {len(prompts) - 1}", f"reply {len(prompts)}"
    # The concepts pass sees the merged summary, and the final answer is both halves.
    assert merge in prompts[-1]
    assert result == f"{merge}\n\n{concepts}"


def test_short_transcript_below_topic_threshold_uses_plain_summary(config, prompts):
    config.topics_min_chars = 1000
    summarize.run("a short transcript", config, "key")

    assert summarize.DETAILED_PROMPT[:40] not in prompts[0]
    assert summarize.SUMMARY_PROMPT[:40] in prompts[0]


def test_transcript_over_topic_threshold_in_one_chunk_uses_detailed_prompt(config, prompts):
    config.chunk_chars = 10_000
    config.topics_min_chars = 10

    result = summarize.run("a transcript long enough to trip the threshold", config, "key")

    assert len(prompts) == 1
    assert summarize.DETAILED_PROMPT[:40] in prompts[0]
    assert result == "reply 1"


def test_long_transcript_over_topic_threshold_gets_topic_breakdown(config, prompts):
    config.topics_min_chars = 10
    transcript = "\n".join(f"paragraph {i} " * 3 for i in range(40))

    result = summarize.run(transcript, config, "key")

    # n chunk calls (using the chunk-detailed prompt), then overview merge, topics
    # merge, then concepts.
    assert len(prompts) > 4
    chunk_prompts, rest = prompts[:-3], prompts[-3:]
    assert all(summarize.CHUNK_DETAILED_PROMPT[:40] in p for p in chunk_prompts)
    assert summarize.OVERVIEW_MERGE_PROMPT[:40] in rest[0]
    assert summarize.TOPICS_MERGE_PROMPT[:40] in rest[1]
    assert summarize.CONCEPTS_PROMPT[:40] in rest[2]

    overview, topics, concepts = (f"reply {len(prompts) - 2}", f"reply {len(prompts) - 1}",
                                   f"reply {len(prompts)}")
    assert overview in rest[2]  # concepts pass sees the merged overview as "summary"
    assert result == f"{overview}\n\n{topics}\n\n{concepts}"


def test_custom_prompt_skips_topic_detection_even_over_threshold(config, prompts):
    config.topics_min_chars = 1
    config.prompt = "TL;DR of {transcript}"

    summarize.run("hello", config, None)

    assert prompts[0] == "TL;DR of hello"


def test_custom_prompt_replaces_the_builtin(config, prompts):
    config.prompt = "TL;DR of {transcript}"

    summarize.run("hello", config, None)

    assert prompts[0] == "TL;DR of hello"


def test_custom_prompt_must_keep_the_placeholder(config, prompts):
    config.prompt = "summarize this please"

    with pytest.raises(ValueError, match=r"\{transcript\} placeholder"):
        summarize.run("hello", config, None)

    assert prompts == []


def test_chunk_returns_text_untouched_when_it_fits():
    assert summarize._chunk("short", 100) == ["short"]


def test_chunk_respects_the_limit_and_keeps_everything():
    text = "\n".join(f"line {i} with some filler words" for i in range(200))

    chunks = summarize._chunk(text, 200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "\n".join(chunks).split() == text.split()


def test_chunk_splits_on_sentences_when_a_paragraph_is_too_long():
    text = " ".join(f"Sentence number {i} runs on for a while." for i in range(50))

    chunks = summarize._chunk(text, 120)

    assert all(len(chunk) <= 120 for chunk in chunks)
    # No sentence was cut in half: every chunk still ends on a full stop.
    assert all(chunk.endswith(".") for chunk in chunks)


def test_chunk_hard_splits_text_with_no_boundaries():
    text = "x" * 500

    chunks = summarize._chunk(text, 100)

    assert chunks == ["x" * 100] * 5


def test_chunk_rejects_a_nonpositive_limit():
    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        summarize._chunk("anything", 0)


class FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def fake_openai(status_code):
    """A stand-in `openai` module whose completions call always fails."""

    def create(**kwargs):
        raise FakeStatusError(status_code)

    module = ModuleType("openai")
    module.APIStatusError = FakeStatusError
    module.OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return module


def test_oversized_request_becomes_an_actionable_error(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", fake_openai(413))

    with pytest.raises(RuntimeError, match="chunk_chars"):
        summarize.run("hello", config, "key")


def test_other_api_errors_are_not_swallowed(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", fake_openai(500))

    with pytest.raises(FakeStatusError):
        summarize.run("hello", config, "key")
