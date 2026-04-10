from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.config import GraphConfig


class LocalLLMAdapter:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.client = self._build_client()

    def _build_client(self):
        if self.config.local_llm_provider != "ollama":
            raise ValueError(
                f"Unsupported LOCAL_LLM_PROVIDER '{self.config.local_llm_provider}'. "
                "Only 'ollama' is supported in phase 0."
            )

        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required for LOCAL_LLM_PROVIDER=ollama."
            ) from exc

        return ChatOllama(
            model=self.config.local_llm_model,
            base_url=self.config.local_llm_base_url,
            temperature=0.0,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return response.content
