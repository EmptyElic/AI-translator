import json
from typing import Annotated, List

import PIL
import requests
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


def _simple_string_field(schema: dict) -> str | None:
    """
    Return the field name if the schema represents a single required string property.
    """
    if not schema:
        return None

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if len(properties) != 1:
        return None

    field_name, field_schema = next(iter(properties.items()))
    field_type = field_schema.get("type")
    if field_type == "string" and field_name in required:
        return field_name

    return None


class OllamaService(BaseService):
    ollama_base_url: Annotated[
        str, "The base url to use for ollama.  No trailing slash."
    ] = "http://localhost:11434"
    ollama_model: Annotated[str, "The model name to use for ollama."] = (
        "llama3.2-vision"
    )

    def process_images(self, images):
        image_bytes = [self.img_to_base64(img) for img in images]
        return image_bytes

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        url = f"{self.ollama_base_url}/api/generate"
        headers = {"Content-Type": "application/json"}

        schema = response_schema.model_json_schema()
        format_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        simple_string_field = _simple_string_field(format_schema)

        image_bytes = self.format_image_for_llm(image)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "images": image_bytes,
        }
        if not simple_string_field:
            payload["format"] = format_schema

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()

            prompt_tokens = response_data.get("prompt_eval_count")
            completion_tokens = response_data.get("eval_count")
            total_tokens = None
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                total_tokens = prompt_tokens + completion_tokens

            if block and total_tokens is not None:
                block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

            data = response_data.get("response", "")
            if not data:
                raise ValueError("Empty response payload from Ollama")

            if simple_string_field:
                parsed = {simple_string_field: data.strip()}
            else:
                parsed = json.loads(data)

            return response_schema.model_validate(parsed)
        except Exception as e:
            logger.warning(f"Ollama inference failed: {e}")

        return {}
