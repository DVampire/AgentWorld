"""OpenAI transcribe model implementation."""

import asyncio
import os
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from pathlib import Path
import io

from pydantic import Field, ConfigDict
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.models.transcribe.base import BaseTranscribeOpenAI, TranscriptionResult
from src.logger import logger


class TranscribeOpenAI(BaseTranscribeOpenAI):
    """OpenAI transcribe model for speech-to-text conversion.
    
    This class provides LangChain-compatible methods for transcribing
    audio files using OpenAI's Whisper API.
    
    Example:
        .. code-block:: python
        
            from src.models.transcribe import TranscribeOpenAI
            from langchain_core.messages import HumanMessage
            
            # Initialize the model
            transcribe = TranscribeOpenAI(
                model_name="whisper-1",
                temperature=0.0,
                language="en"
            )
            
            # Transcribe audio file using invoke
            message = HumanMessage(content="path/to/audio.mp3")
            result = transcribe.invoke([message])
            print(result.content)
            
            # Async transcription
            result = await transcribe.ainvoke([message])
            print(result.content)
    """
    
    model: str = Field(default="whisper-1", description="Model name to use")
    temperature: Optional[float] = Field(default=0.0, description="Sampling temperature")
    language: Optional[str] = Field(default=None, description="Language of the audio")
    prompt: Optional[str] = Field(default=None, description="Optional text to guide the model")
    response_format: Optional[str] = Field(default="json", description="Response format")
    timestamp_granularities: Optional[List[str]] = Field(
        default=None, 
        description="Timestamp granularities for verbose_json format"
    )
    
    # OpenAI specific parameters
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    openai_api_base: Optional[str] = Field(
        default=None,
        description="Base URL for API requests"
    )
    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    request_timeout: Optional[float] = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="Maximum number of retries"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs: Any):
        # Set default values
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        super().__init__(**kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "transcribe_openai"
    
    def _get_audio_duration(self, audio_data: bytes) -> Optional[float]:
        """Get audio duration from audio data."""
        try:
            import librosa
            import io
            
            # Load audio with librosa to get duration
            audio_io = io.BytesIO(audio_data)
            duration = librosa.get_duration(path=audio_io)
            return duration
        except ImportError:
            logger.warning("librosa not available, cannot determine audio duration")
            return None
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return None
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response for LangChain compatibility."""
        # For transcription, we expect a single HumanMessage with audio content
        if len(messages) != 1 or not isinstance(messages[0], HumanMessage):
            raise ValueError("TranscribeOpenAI expects a single HumanMessage with audio content")
        
        message = messages[0]
        audio_content = message.content
        
        # Prepare audio data
        audio_data = self._prepare_audio(audio_content)
        
        # Get audio duration if possible
        duration = self._get_audio_duration(audio_data)
        
        # Perform transcription
        response = self._transcribe_with_retry(audio_data, **kwargs)
        
        # Create transcription result
        transcription_result = TranscriptionResult(
            text=response.get("text", ""),
            language=response.get("language"),
            duration=duration,
            confidence=response.get("confidence"),
            segments=response.get("segments"),
            metadata=response.get("metadata", {}),
        )
        
        # Create LangChain compatible result
        ai_message = AIMessage(
            content=transcription_result.text,
            response_metadata={
                "transcription_result": transcription_result.model_dump(),
                "model_name": self.model,
                "token_usage": response.get("usage", {}),
            }
        )
        
        generation = ChatGeneration(message=ai_message)
        
        # Create ChatResult with usage information for callback handlers
        result = ChatResult(generations=[generation])
        
        # Add usage information to the result for callback handlers
        if response.get("usage"):
            result.llm_output = {"token_usage": response["usage"]}
        
        return result
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response asynchronously for LangChain compatibility."""
        # For transcription, we expect a single HumanMessage with audio content
        if len(messages) != 1 or not isinstance(messages[0], HumanMessage):
            raise ValueError("TranscribeOpenAI expects a single HumanMessage with audio content")
        
        message = messages[0]
        audio_content = message.content
        
        # Prepare audio data
        audio_data = self._prepare_audio(audio_content)
        
        # Get audio duration if possible
        duration = self._get_audio_duration(audio_data)
        
        # Perform transcription
        response = await self._atranscribe_with_retry(audio_data, **kwargs)
        
        # Create transcription result
        transcription_result = TranscriptionResult(
            text=response.get("text", ""),
            language=response.get("language"),
            duration=duration,
            confidence=response.get("confidence"),
            segments=response.get("segments"),
            metadata=response.get("metadata", {}),
        )
        
        # Create LangChain compatible result
        ai_message = AIMessage(
            content=transcription_result.text,
            response_metadata={
                "transcription_result": transcription_result.model_dump(),
                "model_name": self.model,
                "token_usage": response.get("usage", {}),
            }
        )
        
        generation = ChatGeneration(message=ai_message)
        
        # Create ChatResult with usage information for callback handlers
        result = ChatResult(generations=[generation])
        
        # Add usage information to the result for callback handlers
        if response.get("usage"):
            result.llm_output = {"token_usage": response["usage"]}
        
        return result
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream response for LangChain compatibility.
        
        Note: OpenAI Whisper API doesn't support streaming, so this returns
        the complete result as a single chunk.
        """
        result = self._generate(messages, stop, run_manager, **kwargs)
        for generation in result.generations:
            yield generation
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """Stream response asynchronously for LangChain compatibility.
        
        Note: OpenAI Whisper API doesn't support streaming, so this returns
        the complete result as a single chunk.
        """
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        for generation in result.generations:
            yield generation
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model,
            "temperature": self.temperature,
            "language": self.language,
            "response_format": self.response_format,
        }
