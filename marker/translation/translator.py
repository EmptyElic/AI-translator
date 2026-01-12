from __future__ import annotations

from typing import Dict, List, Optional

from rapidfuzz import process
from pydantic import BaseModel

from marker.logger import get_logger
from marker.renderers.markdown import MarkdownOutput
from marker.services import BaseService

logger = get_logger()

# Supported translation targets.
LANGUAGE_REGISTRY: Dict[str, Dict[str, str]] = {
    "ru": {
        "display_name": "Russian",
        "code": "ru",
    },
}

LANGUAGE_ALIASES = {
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
}


def _resolve_language(language: str) -> Dict[str, str]:
    normalized = language.strip().lower()
    if normalized in LANGUAGE_ALIASES:
        key = LANGUAGE_ALIASES[normalized]
        return LANGUAGE_REGISTRY[key]

    match = process.extractOne(
        normalized,
        LANGUAGE_ALIASES.keys(),
        score_cutoff=70,
    )
    if match:
        key = LANGUAGE_ALIASES[match[0]]
        return LANGUAGE_REGISTRY[key]

    raise ValueError(
        f"Translation language '{language}' is not supported. "
        f"Supported languages: {sorted({info['display_name'] for info in LANGUAGE_REGISTRY.values()})}"
    )


def _chunk_text(text: str, max_chunk_chars: int = 2500) -> List[str]:
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


class TranslationResponse(BaseModel):
    translation: str


def _translate_chunk(
    llm_service: BaseService,
    chunk: str,
    target_language: str,
    temperature: float = 0.1,
) -> str:
    if not chunk.strip():
        return chunk

    prompt = (
        "You are a precise technical translator. Convert the following Markdown content "
        f"to {target_language}. Preserve Markdown structure, LaTeX expressions, code blocks, "
        "and inline formatting. Return only the translated Markdown without explanations.\n\n"
        f"Markdown:\n{chunk}"
    )
    response = llm_service(
        prompt=prompt,
        image=None,
        block=None,
        response_schema=TranslationResponse,
        timeout=getattr(llm_service, "timeout", None),
    )
    return response.translation.strip()


def translate_rendered_output(
    rendered: MarkdownOutput,
    target_language: str,
    llm_service: Optional[BaseService] = None,
) -> MarkdownOutput:
    """
    Translate rendered Markdown using the configured LLM service.
    """
    if not isinstance(rendered, MarkdownOutput):
        raise ValueError("Translation is currently only supported for markdown output.")

    if llm_service is None:
        raise RuntimeError(
            "Translation requires an LLM service. Re-run with --use_llm "
            "and ensure an llm_service is configured (for example, GoogleGeminiService)."
        )

    language_info = _resolve_language(target_language)
    chunks = _chunk_text(rendered.markdown)
    translated_segments = [
        _translate_chunk(llm_service, chunk, language_info["display_name"]) for chunk in chunks
    ]
    rendered.markdown = "".join(translated_segments)

    translation_meta = rendered.metadata.setdefault("translation", {})
    translation_meta.update(
        {
            "target_language": language_info["display_name"],
            "translator": llm_service.__class__.__name__,
        }
    )

    logger.info(
        "Translated markdown output to %s using %s",
        language_info["display_name"],
        llm_service.__class__.__name__,
    )

    return rendered
