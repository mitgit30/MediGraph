import logging

from src.graph.config import GraphConfig

logger = logging.getLogger(__name__)


class LocalLLMAdapter:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.client = self._build_client()

    def _build_client(self):
        if self.config.local_llm_provider != "ollama":
            raise ValueError(
                f"Unsupported OLLAMA_LLM_PROVIDER '{self.config.local_llm_provider}'. "
                "Only 'ollama' is supported in phase 0."
            )

        try:
            from ollama import Client
        except ImportError as exc:
            raise ImportError("ollama package is required for OLLAMA_LLM_PROVIDER=ollama.") from exc

        headers = {}
        if self.config.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.config.ollama_api_key}"

        logger.info("Initializing Ollama client with host=%s", self.config.local_llm_base_url)
        return Client(
            host=self.config.local_llm_base_url,
            headers=headers or None,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat(
                model=self.config.local_llm_model,
                messages=messages,
                stream=False,
            )
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            return response["message"]["content"]
        except Exception as exc:
            logger.exception("Ollama text generation failed.")
            raise RuntimeError("Ollama generation failed.") from exc
