import json
import logging
import time

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# How much of a rejected answer / how many validation errors to quote back
# to the model in a retry prompt -- enough to correct from, without letting
# a long bad answer swamp the original instruction.
_MAX_ECHOED_ANSWER_CHARS = 2000
_MAX_ECHOED_ERRORS = 5


class _TransientError(Exception):
    """A failed call worth retrying unchanged: connection error, timeout,
    server error, or a response that wasn't valid JSON."""


class _FatalError(Exception):
    """A failed call that retrying can't fix, e.g. the model isn't pulled
    (Ollama answers 404) or the request itself is malformed (400)."""


class OllamaExtractor:
    """
    A generic structured extractor that forces local Ollama models to return
    JSON matching a specified Pydantic schema, with automatic retry-on-failure
    (self-healing) using the validation errors.
    """
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "gemma3:12b",
        timeout: int = 90,
        options: dict | None = None,
    ):
        """
        Args:
            ollama_url: Ollama's /api/generate endpoint.
            model: Any model you have pulled into Ollama.
            timeout: Seconds to wait for one call.
            options: Ollama model options (temperature, num_ctx, seed, ...).
                Defaults to {"temperature": 0}: extraction should return the
                same answer for the same input, not a creative variation.
        """
        self.url = ollama_url
        self.model = model
        self.timeout = timeout
        self.options = {"temperature": 0} if options is None else options

    def _call_ollama(self, prompt: str, format_schema: dict | str) -> dict | list:
        """One call to Ollama, returning the parsed JSON answer. Raises
        _TransientError or _FatalError instead of returning None, so the
        caller can tell "try again" from "stop"."""
        try:
            res = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "format": format_schema,
                    "stream": False,
                    "prompt": prompt,
                    "options": self.options,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise _TransientError(f"{type(e).__name__}: {e}") from e

        if res.status_code >= 400:
            try:
                detail = res.json().get("error") or res.text
            except ValueError:
                detail = res.text
            message = f"HTTP {res.status_code} from Ollama: {detail}"
            if 400 <= res.status_code < 500:
                raise _FatalError(message)
            raise _TransientError(message)

        try:
            return json.loads(res.json()["response"])
        except (KeyError, TypeError, ValueError) as e:
            raise _TransientError(f"unparseable response ({type(e).__name__}): {e}") from e

    @staticmethod
    def _format_schema(schema: type[BaseModel], many: bool) -> dict:
        """JSON schema for Ollama's `format` parameter. For a list, the item
        schema's `$defs` (present whenever the model nests other models) must
        move to the root: `$ref`s like "#/$defs/Address" resolve from the
        document root, so leaving `$defs` inside "items" breaks every one."""
        item_schema = schema.model_json_schema()
        if not many:
            return item_schema
        defs = item_schema.pop("$defs", None)
        array_schema = {"type": "array", "items": item_schema}
        if defs:
            array_schema["$defs"] = defs
        return array_schema

    @staticmethod
    def _normalize_extracted_items(raw: dict | list) -> list:
        """Defensive fallback for the 'many' case if the model wraps the output."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("items", "data", "results"):
                if isinstance(raw.get(key), list):
                    return raw[key]
            return [raw]
        return []

    @staticmethod
    def _validate_items(schema: type[BaseModel], items: list) -> tuple[list[BaseModel], list[str]]:
        """Validate each item; returns (valid items, one error message per invalid item)."""
        validated, errors = [], []
        for index, item in enumerate(items):
            try:
                validated.append(schema.model_validate(item))
            except (ValidationError, TypeError, AttributeError) as e:
                errors.append(f"item {index}: {e}")
        return validated, errors

    @staticmethod
    def _retry_prompt(prompt: str, previous_answer, problem: str) -> str:
        answer = json.dumps(previous_answer, ensure_ascii=False)
        if len(answer) > _MAX_ECHOED_ANSWER_CHARS:
            answer = answer[:_MAX_ECHOED_ANSWER_CHARS] + " ...(truncated)"
        return (
            f"{prompt}\n\n"
            f"Your previous answer was:\n{answer}\n\n"
            f"It was invalid: {problem}\n"
            f"Return corrected JSON only."
        )

    def extract(
        self,
        prompt: str,
        schema: type[BaseModel],
        many: bool = False,
        max_retries: int = 1,
        network_retry_delay: float = 2.0,
    ) -> BaseModel | list[BaseModel] | None:
        """
        Extracts structured data matching the Pydantic schema.
        If validation fails, it feeds the previous answer and the validation
        errors back to the model for a retry. If the Ollama call itself fails
        transiently (timeout, connection refused, server error, unparseable
        response), it retries the same prompt unchanged; a client error such as
        a model that isn't pulled (HTTP 4xx) is not retried.

        Args:
            prompt: The instruction and text to extract from.
            schema: The Pydantic BaseModel to validate against.
            many: If True, expects a list of objects matching the schema.
            max_retries: Number of retries on validation or transient failure.
            network_retry_delay: Seconds to wait before retrying after a transient failure.

        Returns:
            A validated model instance (many=False), or a list of them
            (many=True) -- an empty list when the text genuinely contains no
            items. None if extraction fails. With many=True, if some items
            still fail validation after all retries, the valid ones are
            returned and the rest are logged.
        """
        format_schema = self._format_schema(schema, many)

        current_prompt = prompt
        for attempt in range(max_retries + 1):
            last_attempt = attempt == max_retries
            try:
                raw = self._call_ollama(current_prompt, format_schema)
            except _FatalError as e:
                logger.warning("Ollama call failed, not retrying: %s", e)
                return None
            except _TransientError as e:
                logger.warning("Ollama call failed (attempt %d of %d): %s", attempt + 1, max_retries + 1, e)
                if last_attempt:
                    return None
                time.sleep(network_retry_delay)
                continue

            if many:
                if not isinstance(raw, (list, dict)):
                    # A bare string/number isn't "no items" -- it's a malformed answer.
                    if last_attempt:
                        logger.warning("Giving up: expected a JSON array, got %s", type(raw).__name__)
                        return None
                    current_prompt = self._retry_prompt(
                        prompt, raw, f"expected a JSON array of objects, got {type(raw).__name__}"
                    )
                    continue
                items = self._normalize_extracted_items(raw)
                if not items:
                    return []  # a valid answer: the text has no matching items
                validated, errors = self._validate_items(schema, items)
                if not errors:
                    return validated
                if last_attempt:
                    for error in errors:
                        logger.warning("Dropping invalid item after %d attempt(s): %s", attempt + 1, error)
                    return validated or None
                current_prompt = self._retry_prompt(
                    prompt, raw, f"{len(errors)} of {len(items)} item(s) failed validation:\n"
                    + "\n".join(errors[:_MAX_ECHOED_ERRORS])
                )
                continue

            try:
                return schema.model_validate(raw)
            except ValidationError as e:
                if last_attempt:
                    logger.warning("Giving up after %d failed validation attempt(s): %s", attempt + 1, e)
                    return None
                current_prompt = self._retry_prompt(prompt, raw, str(e))

        return None
