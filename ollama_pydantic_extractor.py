import json
import requests
from pydantic import BaseModel, ValidationError

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
            return json.loads(res.json()["response"])
        except Exception:
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

    def extract(self, prompt: str, schema: type[BaseModel], many: bool = False, max_retries: int = 1):
        """
        Extracts structured data matching the Pydantic schema.
        If validation fails, it feeds the error back to the model for a retry.
        
        Args:
            prompt: The instruction and text to extract from.
            schema: The Pydantic BaseModel to validate against.
            many: If True, expects a list of objects matching the schema.
            max_retries: Number of times to retry on validation failure.
        
        Returns:
            A validated Pydantic model instance, a list of instances, or None if extraction fails.
        """
        item_schema = schema.model_json_schema()
        format_schema = {"type": "array", "items": item_schema} if many else item_schema
        
        raw = self._extract_via_ollama(prompt, format_schema)
        
        for attempt in range(max_retries + 1):
            if raw is None:
                return None
            try:
                if many:
                    items = self._normalize_extracted_items(raw)
                    return [schema.model_validate(item) for item in items]
                return schema.model_validate(raw)
            except ValidationError as e:
                if attempt == max_retries:
                    return None
                
                # Self-healing retry: append the validation error to the prompt
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous answer was invalid: {e}\n"
                    f"Return corrected JSON only."
                )
                raw = self._extract_via_ollama(retry_prompt, format_schema)
                
        return None
