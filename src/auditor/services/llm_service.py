from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, Type, Any, Dict


class LLMService:
    def __init__(self, api_key: str, model: str = "deepseek-ai/deepseek-v3.1", temperature: float = 0.7):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client: Optional[ChatNVIDIA] = None

    @property
    def client(self) -> ChatNVIDIA:
        if self._client is None:
            self._client = ChatNVIDIA(
                model=self._model,
                nvidia_api_key=self._api_key,
                temperature=self._temperature
            )
        return self._client

    def _get_client(self, temperature: Optional[float] = None) -> ChatNVIDIA:
        if temperature is not None:
            return ChatNVIDIA(
                model=self._model,
                nvidia_api_key=self._api_key,
                temperature=temperature
            )
        return self.client

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))

        client = self._get_client(temperature)
        response = client.invoke(messages)
        return response.content

    def complete_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        schema: Optional[Type[Any]] = None
    ) -> Dict[str, Any]:
        parser = JsonOutputParser(pydantic_schema=schema) if schema else JsonOutputParser()
        
        format_instructions = parser.get_format_instructions()
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=f"{prompt}\n\n{format_instructions}"))

        client = self._get_client(temperature)
        response = client.invoke(messages)
        return parser.parse(response.content)