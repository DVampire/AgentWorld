"""Base classes for transcribe models."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from pathlib import Path
import io

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.logger import logger


class TranscriptionResult(BaseModel):
    """Result of a transcription operation."""
    
    text: str = Field(description="The transcribed text")
    language: Optional[str] = Field(default=None, description="Detected language")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed segments")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseTranscribeModel(BaseChatModel, ABC):
    """Base class for transcribe models."""
    
    model_name: str = Field(default="whisper-1", description="Model name to use")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    language: Optional[str] = Field(default=None, description="Language of the audio")
    prompt: Optional[str] = Field(default=None, description="Optional text to guide the model")
    response_format: Optional[str] = Field(default="json", description="Response format")
    timestamp_granularities: Optional[List[str]] = Field(default=None, description="Timestamp granularities")
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "transcribe"
    
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response for LangChain compatibility."""
        pass
    
    @abstractmethod
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response asynchronously for LangChain compatibility."""
        pass
    
    def _prepare_audio(
        self, 
        audio: Union[str, Path, bytes, io.BytesIO]
    ) -> bytes:
        """Prepare audio data for API call."""
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            with open(audio_path, "rb") as f:
                return f.read()
        elif isinstance(audio, bytes):
            return audio
        elif isinstance(audio, io.BytesIO):
            return audio.getvalue()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
    
    def _get_invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Get parameters for model invocation."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "language": self.language,
            "prompt": self.prompt,
            "response_format": self.response_format,
            "timestamp_granularities": self.timestamp_granularities,
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        # Add any additional kwargs
        params.update(kwargs)
        return params
    
    def _create_transcription_result(
        self, 
        response: Dict[str, Any],
        audio_duration: Optional[float] = None
    ) -> TranscriptionResult:
        """Create TranscriptionResult from API response."""
        return TranscriptionResult(
            text=response.get("text", ""),
            language=response.get("language"),
            duration=audio_duration,
            confidence=response.get("confidence"),
            segments=response.get("segments"),
            metadata=response.get("metadata", {}),
        )


class BaseTranscribeOpenAI(BaseTranscribeModel):
    """Base class for OpenAI transcribe models."""
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_api_base: Optional[str] = Field(default=None, description="Base URL for API requests")
    openai_organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    request_timeout: Optional[float] = Field(default=None, description="Request timeout")
    max_retries: Optional[int] = Field(default=None, description="Maximum number of retries")
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._setup_openai_client()
    
    def _setup_openai_client(self):
        """Setup OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
                organization=self.openai_organization,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            self._async_client = openai.AsyncOpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
                organization=self.openai_organization,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it with: pip install openai"
            )
    
    def _get_request_params(self, **kwargs: Any) -> tuple[str, Dict[str, Any]]:
        """Get parameters for OpenAI API request."""
        params = {
            "temperature": self.temperature,
            "language": self.language,
            "prompt": self.prompt,
            "response_format": self.response_format,
            "timestamp_granularities": self.timestamp_granularities,
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        # Add any additional kwargs
        params.update(kwargs)
        return self.model, params
    
    async def _atranscribe_with_retry(
        self,
        audio: bytes,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio with retry logic."""
        model, params = self._get_request_params(**kwargs)
        
        try:
            response = await self._async_client.audio.transcriptions.create(
                model=model,
                file=("audio", audio, "audio/mpeg"),
                **params
            )
            return response.model_dump() if hasattr(response, 'model_dump') else response
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_with_retry(
        self,
        audio: bytes,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio with retry logic."""
        model, params = self._get_request_params(**kwargs)
        
        try:
            response = self._client.audio.transcriptions.create(
                model=model,
                file=("audio", audio, "audio/mpeg"),
                **params
            )
            return response.model_dump() if hasattr(response, 'model_dump') else response
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
