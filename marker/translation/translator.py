from __future__ import annotations

import re
from textwrap import dedent
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
        "native_name": "русский",
        "code": "ru",
    },
}

PROMPT_TEMPLATE = dedent(
    """
    You are a precise technical translator.
    Convert the Markdown content delimited by <MARKDOWN> and </MARKDOWN> into {language_display} ({language_native}).
    STRICT RULES:
    - Preserve Markdown structure, LaTeX math, code fences, HTML spans/anchors, metadata.
    - Do NOT translate inside fenced code blocks, LaTeX, HTML attributes, identifiers.
    - Output ONLY the translated Markdown, no explanations, greetings, or confirmations.
    - Do not wrap the answer in XML/JSON or repeat the prompt.

    <MARKDOWN>
    {content}
    </MARKDOWN>
    """
).strip()

MAX_TRANSLATION_CHARS = 1200
MAX_TRANSLATION_ATTEMPTS = 4
MAX_POST_EDIT_ATTEMPTS = 2
MIN_CYRILLIC_RATIO = 0.25
MIN_CYRILLIC_ABSOLUTE = 25
ENABLE_TRANSLATION_POST_EDIT = True

# NOISE_PREFIXES = [
#     "привет",
#     "ок",
#     "okay",
#     "ok.",
#     "ok!",
#     "выполнено",
#     "готово",
#     "here is the translated content",
#     "please provide",
#     "scrolling",
#     "ниже",
#     "ниже мы",
#     "перевод:",
#     "translation:",
#     "the translated text is",
# ]

# NOISE_REGEX = re.compile(
#     r"^(?:\s*(?:#+\s*)?(?:"
#     + "|".join(re.escape(prefix) for prefix in NOISE_PREFIXES)
#     + r")[:!,. ]*)",
#     re.IGNORECASE,
# )

LANGUAGE_ALIASES = {
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
}

CODE_FENCE_PATTERN = re.compile(r"^\s*`{3,}")


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


def _chunk_text(text: str, max_chunk_chars: int = MAX_TRANSLATION_CHARS) -> List[str]:
    """
    Split markdown into translation-friendly chunks while preserving code fences
    and structural boundaries (headings/blank lines) whenever possible.
    """

    if not text:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    inside_code_fence = False
    fence_delimiter = ""

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append("".join(current))
            current = []
            current_len = 0

    for line in lines:
        stripped = line.strip()

        if CODE_FENCE_PATTERN.match(stripped):
            if not inside_code_fence:
                inside_code_fence = True
                fence_delimiter = stripped.split()[0]
            elif not fence_delimiter or stripped.startswith(fence_delimiter):
                inside_code_fence = False
                fence_delimiter = ""

        def needs_split() -> bool:
            if not current or inside_code_fence:
                return False
            if current_len + len(line) > max_chunk_chars:
                return True
            if not stripped and current_len >= int(max_chunk_chars * 0.8):
                return True
            if stripped.startswith("#") and current_len >= int(max_chunk_chars * 0.6):
                return True
            return False

        if needs_split():
            flush_current()

        current.append(line)
        current_len += len(line)

        # Extreme fallback: if a single fenced block wildly exceeds max size,
        # allow splitting after closing fence to avoid unbounded chunks.
        if (
            not inside_code_fence
            and current_len >= max_chunk_chars * 2
        ):
            flush_current()

    flush_current()

    return chunks or [text]


class TranslationResponse(BaseModel):
    translation: str


class PostEditResponse(BaseModel):
    corrected_markdown: str


POST_EDIT_PROMPT_TEMPLATE = dedent(
    """
    You are a meticulous editor.
    Polish the Markdown between <MARKDOWN> and </MARKDOWN> which is already in {language_display}.
    Fix spacing, duplicated fragments, tables, leftover foreign words, punctuation, and obvious hallucinated artifacts,
    but DO NOT delete sentences, do not summarize, and do not remove math, tables, links, or HTML tags.
    Preserve structure exactly and return only the corrected Markdown.

    <MARKDOWN>
    {content}
    </MARKDOWN>
    """
).strip()


def _clean_translated_output(text: str) -> str:
    """
    Remove residual prompt echoes or injected instructions from the translated output.
    """
    if not text:
        return text

    cleaned = text.replace("<MARKDOWN>", "").replace("</MARKDOWN>", "").strip()

    # cleaned = NOISE_REGEX.sub("", cleaned).lstrip()
    # lines = cleaned.splitlines()
    # while lines and not lines[0].strip():
    #     lines.pop(0)
    # cleaned = "\n".join(lines)

    markers = [
        "you are a precise technical translator",
        "convert the following markdown",
        "return only the translated markdown",
        "below is the markdown content",
    ]
    lowered = cleaned.lower()
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].rstrip()
            lowered = cleaned.lower()

    return cleaned


def _is_cyrillic_char(ch: str) -> bool:
    if ch in ("ё", "Ё"):
        return True
    codepoint = ord(ch)
    return 0x0400 <= codepoint <= 0x052F


def _translation_is_valid(original: str, translated: str) -> bool:
    if not translated.strip():
        return False

    lower = translated.strip().lower()
    if lower.startswith(("please provide", "here is the", "i'm sorry")):
        return False

    alpha_count = sum(1 for ch in translated if ch.isalpha())
    cyrillic_count = sum(1 for ch in translated if _is_cyrillic_char(ch))

    if alpha_count == 0:
        # Mostly equations / symbols. Accept as-is.
        return True

    if cyrillic_count >= MIN_CYRILLIC_ABSOLUTE:
        return True

    ratio = cyrillic_count / max(alpha_count, 1)
    return ratio >= MIN_CYRILLIC_RATIO


def _translate_chunk(
    llm_service: BaseService,
    chunk: str,
    language_info: Dict[str, str],
    temperature: float = 0.1,
    chunk_id: int | None = None,
) -> str:
    if not chunk.strip():
        return chunk

    chunk_label = f"#{chunk_id + 1}" if chunk_id is not None else "n/a"

    prompt_base = PROMPT_TEMPLATE.format(
        language_display=language_info["display_name"],
        language_native=language_info.get("native_name", language_info["display_name"]),
        content=chunk,
    )

    for attempt in range(1, MAX_TRANSLATION_ATTEMPTS + 1):
        prompt = prompt_base
        if attempt > 1:
            prompt += (
                "\n\nREMINDER: Output ONLY the translated Markdown in "
                f"{language_info['display_name']} with no extra comments."
            )

        logger.info(
            "Translation chunk %s attempt %s prompt:\n%s",
            chunk_label,
            attempt,
            prompt,
        )

        response = llm_service(
            prompt=prompt,
            image=None,
            block=None,
            response_schema=TranslationResponse,
            timeout=getattr(llm_service, "timeout", None),
        )

        logger.info(
            "Translation chunk %s attempt %s raw response: %s",
            chunk_label,
            attempt,
            response,
        )

        if isinstance(response, dict):
            translation_text = response.get("translation", "")
        else:
            translation_text = response.translation

        if not isinstance(translation_text, str):
            logger.warning("LLM translation payload missing text. Attempt %s", attempt)
            continue

        cleaned = _clean_translated_output(translation_text.strip())
        if _translation_is_valid(chunk, cleaned):
            return cleaned

        logger.warning(
            "LLM translation looked invalid on attempt %s/%s; retrying.",
            attempt,
            MAX_TRANSLATION_ATTEMPTS,
        )

    logger.error("LLM translation failed; returning original chunk.")
    return chunk


def _post_edit_chunk(
    llm_service: BaseService,
    chunk: str,
    language_info: Dict[str, str],
    chunk_id: int | None = None,
) -> str:
    if not chunk.strip():
        return chunk

    chunk_label = f"#{chunk_id + 1}" if chunk_id is not None else "n/a"
    prompt = POST_EDIT_PROMPT_TEMPLATE.format(
        language_display=language_info["display_name"], content=chunk
    )

    for attempt in range(1, MAX_POST_EDIT_ATTEMPTS + 1):
        logger.info(
            "Post-edit chunk %s attempt %s prompt:\n%s",
            chunk_label,
            attempt,
            prompt,
        )
        response = llm_service(
            prompt=prompt,
            image=None,
            block=None,
            response_schema=PostEditResponse,
            timeout=getattr(llm_service, "timeout", None),
        )
        logger.info(
            "Post-edit chunk %s attempt %s raw response: %s",
            chunk_label,
            attempt,
            response,
        )

        if isinstance(response, dict):
            corrected_text = response.get("corrected_markdown", "")
        else:
            corrected_text = response.corrected_markdown

        if not isinstance(corrected_text, str):
            logger.warning("Post-edit payload missing text. Attempt %s", attempt)
            continue

        corrected_text = corrected_text.strip()
        if not corrected_text:
            continue

        original_len = len(chunk.strip())
        if original_len and len(corrected_text) < original_len * 0.5:
            logger.warning(
                "Post-edit chunk %s attempt %s looked truncated; skipping replacement.",
                chunk_label,
                attempt,
            )
            continue

        return corrected_text

    logger.warning("Post-edit failed for chunk %s; keeping original chunk.", chunk_label)
    return chunk


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
        _translate_chunk(llm_service, chunk, language_info, chunk_id=idx)
        for idx, chunk in enumerate(chunks)
    ]
    post_processed_segments = translated_segments
    if ENABLE_TRANSLATION_POST_EDIT:
        post_processed_segments = [
            _post_edit_chunk(llm_service, segment, language_info, chunk_id=idx)
            for idx, segment in enumerate(translated_segments)
        ]

    rendered.markdown = "".join(post_processed_segments)

    translation_meta = rendered.metadata.setdefault("translation", {})
    translation_meta.update(
        {
            "target_language": language_info["display_name"],
            "translator": llm_service.__class__.__name__,
            "post_edit_enabled": ENABLE_TRANSLATION_POST_EDIT,
        }
    )

    logger.info(
        "Translated markdown output to %s using %s",
        language_info["display_name"],
        llm_service.__class__.__name__,
    )

    return rendered
