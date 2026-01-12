from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from rapidfuzz import process
from transformers import pipeline

from marker.logger import get_logger
from marker.renderers.markdown import MarkdownOutput

logger = get_logger()

# Supported translation targets and the HuggingFace models backing them.
LANGUAGE_MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "ru": {
        "model_name": "Helsinki-NLP/opus-mt-en-ru",
        "display_name": "Russian",
    },
}

# Common aliases for supported languages.
LANGUAGE_ALIASES = {
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
}


def _resolve_language(language: str) -> Dict[str, str]:
    """Map a user provided language string to a supported registry entry."""
    normalized = language.strip().lower()
    if normalized in LANGUAGE_ALIASES:
        key = LANGUAGE_ALIASES[normalized]
        return LANGUAGE_MODEL_REGISTRY[key]

    # Fall back to a fuzzy match to handle typos (e.g., "Russin").
    match = process.extractOne(
        normalized,
        LANGUAGE_ALIASES.keys(),
        score_cutoff=70,
    )
    if match:
        key = LANGUAGE_ALIASES[match[0]]
        return LANGUAGE_MODEL_REGISTRY[key]

    raise ValueError(
        f"Translation language '{language}' is not supported. "
        f"Supported languages: {sorted({info['display_name'] for info in LANGUAGE_MODEL_REGISTRY.values()})}"
    )


def _chunk_text(text: str, max_chunk_chars: int = 3500) -> List[str]:
    """Split text into chunks that respect paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for idx, paragraph in enumerate(paragraphs):
        if idx < len(paragraphs) - 1:
            paragraph += "\n\n"

        if len(current) + len(paragraph) > max_chunk_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current += paragraph

    if current:
        chunks.append(current)

    return chunks or [text]


@lru_cache(maxsize=4)
def _get_pipeline(model_name: str):
    """Cache translation pipelines to avoid re-loading models."""
    return pipeline("translation", model=model_name, tokenizer=model_name)


def _translate_text(text: str, model_name: str) -> str:
    if not text.strip():
        return text

    translator = _get_pipeline(model_name)
    chunks = _chunk_text(text)
    translated_segments: List[str] = []
    try:
        responses = translator(chunks, truncation=True, max_length=2048)
    except Exception as exc:  # pragma: no cover - transformers raises various errors
        logger.error(f"Translation failed using model {model_name}: {exc}")
        raise

    # Transformers pipeline returns a dict when a single string is passed and a list otherwise.
    if isinstance(responses, dict):
        responses = [responses]

    # Ensure we align counts even if pipeline merges responses.
    if len(responses) != len(chunks):
        logger.warning(
            "Unexpected translation response length. Falling back to sequential processing."
        )
        translated_segments = [
            translator(chunk, truncation=True, max_length=2048)[0]["translation_text"]
            if chunk.strip()
            else chunk
            for chunk in chunks
        ]
    else:
        translated_segments = [
            response.get("translation_text", "") if chunks[idx].strip() else chunks[idx]
            for idx, response in enumerate(responses)
        ]

    return "".join(translated_segments)


def translate_rendered_output(rendered: MarkdownOutput, target_language: str) -> MarkdownOutput:
    """
    Translate the textual content of a rendered Markdown output into the target language.

    Currently only Markdown output is supported.
    """
    if not isinstance(rendered, MarkdownOutput):
        raise ValueError("Translation is currently only supported for markdown output.")

    language_info = _resolve_language(target_language)
    translated_markdown = _translate_text(rendered.markdown, language_info["model_name"])
    rendered.markdown = translated_markdown

    translation_meta = rendered.metadata.setdefault("translation", {})
    translation_meta.update(
        {
            "target_language": language_info["display_name"],
            "model_name": language_info["model_name"],
        }
    )

    logger.info(
        "Translated markdown output to %s using %s",
        language_info["display_name"],
        language_info["model_name"],
    )

    return rendered
