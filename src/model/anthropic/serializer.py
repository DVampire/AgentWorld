from typing import overload, Any, List, Union, Optional
import base64
import os

from src.message.types import (
    AssistantMessage,
    ContentPartImage,
    ContentPartRefusal,
    ContentPartText,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
)
from src.utils import assemble_project_path, decode_file_base64


class AnthropicChatSerializer:
    """
    Serializer for converting between custom message types and Anthropic messages API format.
    
    Anthropic API format:
    - system: string (top-level field, not in messages)
    - messages: list of {"role": "user"|"assistant", "content": [...]}
    - content for user: list of {"type": "text"|"image", ...}
    - content for assistant: list of {"type": "text"|"tool_use", ...}
    - images: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    """

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> dict[str, Any]:
        return {"type": "text", "text": part.text}

    @staticmethod
    def _normalize_media_type(media_type: str) -> str:
        """Normalize media type to Anthropic-supported formats.
        
        Anthropic only supports: 'image/jpeg', 'image/png', 'image/gif', 'image/webp'
        """
        media_type = media_type.lower().strip()
        
        # Map common variations to supported types
        if media_type in ['image/jpeg', 'image/jpg']:
            return 'image/jpeg'
        elif media_type == 'image/png':
            return 'image/png'
        elif media_type == 'image/gif':
            return 'image/gif'
        elif media_type == 'image/webp':
            return 'image/webp'
        else:
            # Default to jpeg for unknown types
            return 'image/jpeg'

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> dict[str, Any]:
        """Serialize image content part for Anthropic API.
        
        Anthropic expects: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        Anthropic only supports: 'image/jpeg', 'image/png', 'image/gif', 'image/webp'
        """
        image_url = part.image_url.url
        
        # Handle data URLs (base64 encoded)
        if image_url.startswith("data:"):
            # Extract media type and base64 data from data URL
            # Format: data:image/jpeg;base64,<base64_data>
            header, data = image_url.split(",", 1)
            media_type = "image/jpeg"  # default
            if "image/" in header:
                extracted_type = header.split("image/")[1].split(";")[0]
                media_type = AnthropicChatSerializer._normalize_media_type(f"image/{extracted_type}")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                }
            }
        elif image_url.startswith("file://"):
            # File path - read and encode to base64
            file_path = image_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # Read file and encode to base64
                with open(file_path, "rb") as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                # Guess media type from file extension
                import mimetypes
                guessed_type, _ = mimetypes.guess_type(file_path)
                if not guessed_type or not guessed_type.startswith("image/"):
                    media_type = "image/jpeg"  # default
                else:
                    media_type = AnthropicChatSerializer._normalize_media_type(guessed_type)
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                }
        elif os.path.exists(image_url):
            # Direct file path
            with open(image_url, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(image_url)
            if not guessed_type or not guessed_type.startswith("image/"):
                media_type = "image/jpeg"  # default
            else:
                media_type = AnthropicChatSerializer._normalize_media_type(guessed_type)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }
            }
        elif os.path.exists(assemble_project_path(image_url)):
            # Relative file path
            file_path = assemble_project_path(image_url)
            with open(file_path, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(file_path)
            if not guessed_type or not guessed_type.startswith("image/"):
                media_type = "image/jpeg"  # default
            else:
                media_type = AnthropicChatSerializer._normalize_media_type(guessed_type)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }
            }
        else:
            # URL - try to decode if it's base64, otherwise raise error
            # Anthropic doesn't support direct URLs, only base64
            raise ValueError(f"Anthropic API only supports base64-encoded images or local files. Got: {image_url}")

    @staticmethod
    def _serialize_user_content(
        content: Union[str, List[Union[ContentPartText, ContentPartImage]]],
    ) -> List[dict[str, Any]]:
        """Serialize content for user messages (text and images allowed).
        
        Anthropic requires content to always be an array, even for text-only messages.
        """
        serialized_parts: List[dict[str, Any]] = []
        
        if isinstance(content, str):
            # Convert string to text content block
            serialized_parts.append({"type": "text", "text": content})
        else:
            # Process content parts
            for part in content:
                if part.type == 'text':
                    serialized_parts.append(AnthropicChatSerializer._serialize_content_part_text(part))
                elif part.type == 'image_url':
                    serialized_parts.append(AnthropicChatSerializer._serialize_content_part_image(part))
        
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: Optional[Union[str, List[ContentPartText]]],
    ) -> List[dict[str, Any]]:
        """Serialize content for assistant messages (text only, tool_use handled separately).
        
        Anthropic requires content to always be an array, even for text-only messages.
        """
        serialized_parts: List[dict[str, Any]] = []
        
        if content is None:
            return serialized_parts
        
        if isinstance(content, str):
            # Convert string to text content block
            serialized_parts.append({"type": "text", "text": content})
        else:
            # Process content parts
            for part in content:
                if part.type == 'text':
                    serialized_parts.append(AnthropicChatSerializer._serialize_content_part_text(part))
        
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
        """Serialize tool call for Anthropic API.
        
        Anthropic expects: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        """
        import json
        try:
            input_data = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
        except json.JSONDecodeError:
            input_data = {}
        
        return {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": input_data,
        }

    @overload
    @staticmethod
    def serialize(message: HumanMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> dict[str, Any]: ...

    @staticmethod
    def serialize(message: Message) -> dict[str, Any]:
        """Serialize a custom message to an Anthropic message format."""
        if isinstance(message, HumanMessage):
            content = AnthropicChatSerializer._serialize_user_content(message.content)
            result: dict[str, Any] = {
                'role': 'user',
                'content': content,
            }
            return result

        elif isinstance(message, SystemMessage):
            # System messages are handled separately (top-level system field)
            # Return None or empty dict to indicate it should be extracted
            content = message.content
            if isinstance(content, str):
                return {'role': 'system', 'content': content}
            elif isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, ContentPartText):
                        text_parts.append(part.text)
                return {'role': 'system', 'content': ' '.join(text_parts)}
            else:
                return {'role': 'system', 'content': str(content)}

        elif isinstance(message, AssistantMessage):
            content_parts = AnthropicChatSerializer._serialize_assistant_content(message.content)
            result: dict[str, Any] = {'role': 'assistant'}
            
            # Add tool calls to content array
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    content_parts.append(AnthropicChatSerializer._serialize_tool_call(tool_call))
            
            # Content is always an array (may be empty)
            result['content'] = content_parts
            
            return result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: List[Message]) -> tuple[Optional[str], List[dict[str, Any]]]:
        """
        Serialize messages to Anthropic format.
        
        Returns:
            Tuple of (system_message, messages_list)
            system_message: Optional string for system prompt (extracted from SystemMessage)
            messages_list: List of message dicts (excluding SystemMessage)
        """
        system_message = None
        anthropic_messages: List[dict[str, Any]] = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # Extract system message
                serialized = AnthropicChatSerializer.serialize(message)
                if serialized.get('content'):
                    if system_message is None:
                        system_message = serialized['content']
                    else:
                        system_message += "\n" + serialized['content']
            else:
                # Serialize user/assistant messages
                anthropic_messages.append(AnthropicChatSerializer.serialize(message))
        
        return system_message, anthropic_messages

