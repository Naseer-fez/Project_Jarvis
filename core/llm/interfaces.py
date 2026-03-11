from typing import Any, Protocol

class LLMProvider(Protocol):
    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        task_type: str = "chat",
        keep_think: bool = False,
    ) -> str: ...

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        task_type: str = "planning",
    ) -> dict[str, Any] | None: ...

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> str: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> str: ...
