import json
import logging
import time

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OllamaExtractor:
    """
    A generic structured extractor that forces local Ollama models to return
    JSON matching a specified Pydantic schema, with automatic retry-on-failure
    (self-healing) using the validation errors.
    """
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "gemma3:12b", timeout: int = 90):
        self.url = ollama_url
        self.model = model
        self.timeout = timeout

    def _extract_via_ollama(self, prompt: str, format_schema: dict | str = "json") -> dict | list | None:
        """Single call to local Ollama. `format_schema` should ideally be a JSON schema."""
        try:
            res = requests.post(
                self.url,
                json={"model": self.model, "format": format_schema, "stream": False, "prompt": prompt},
                timeout=self.timeout,
            )
            res.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Ollama request failed (%s): %s", type(e).__name__, e)
            return None

        try:
            body = res.json()
            return json.loads(body["response"])
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Ollama returned an unparseable response (%s): %s", type(e).__name__, e)
            return None

    def _normalize_extracted_items(self, raw: dict | list | None) -> list:
        """Defensive fallback for the 'many' case if the model wraps the output."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("items", "data", "results"):
                if isinstance(raw.get(key), list):
                    return raw[key]
            return [raw]
        return []

    def _validate_items(self, schema: type[BaseModel], items: list) -> list[BaseModel]:
        """Validate each item, skipping (and logging) any that aren't even shaped like the schema."""
        validated = []
        for item in items:
            try:
                validated.append(schema.model_validate(item))
            except (ValidationError, TypeError, AttributeError) as e:
                logger.warning("Skipping unvalidatable item %r: %s", item, e)
        return validated

    def extract(
        self,
        prompt: str,
        schema: type[BaseModel],
        many: bool = False,
        max_retries: int = 1,
        network_retry_delay: float = 2.0,
    ):
        """
        Extracts structured data matching the Pydantic schema.
        If validation fails, it feeds the error back to the model for a retry.
        If the Ollama call itself fails (timeout, connection refused, bad response),
        it retries the same prompt unchanged.

        Args:
            prompt: The instruction and text to extract from.
            schema: The Pydantic BaseModel to validate against.
            many: If True, expects a list of objects matching the schema.
            max_retries: Number of times to retry on validation failure or network/parse failure.
            network_retry_delay: Seconds to wait before retrying after a network/parse failure.

        Returns:
            A validated Pydantic model instance, a list of instances, or None if extraction fails.
        """
        item_schema = schema.model_json_schema()
        format_schema = {"type": "array", "items": item_schema} if many else item_schema

        current_prompt = prompt
        for attempt in range(max_retries + 1):
            raw = self._extract_via_ollama(current_prompt, format_schema)

            if raw is None:
                if attempt == max_retries:
                    logger.warning("Giving up after %d failed Ollama call(s).", attempt + 1)
                    return None
                time.sleep(network_retry_delay)
                continue

            if many:
                items = self._normalize_extracted_items(raw)
                validated = self._validate_items(schema, items)
                if validated or attempt == max_retries:
                    return validated
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous answer contained no items matching the required schema.\n"
                    f"Return corrected JSON only."
                )
                continue

            try:
                return schema.model_validate(raw)
            except ValidationError as e:
                if attempt == max_retries:
                    logger.warning("Giving up after %d failed validation attempt(s): %s", attempt + 1, e)
                    return None
                # Self-healing retry: append the validation error to the prompt
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous answer was invalid: {e}\n"
                    f"Return corrected JSON only."
                )

        return None
