from typing import List, Dict, Any, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class MessageManager():
    def __init__(self, 
                 api_type: str = "chat/completions",
                 model: str = "o3"):
        self.api_type = api_type
        self.model = model
        
    def __call__(self, messages: List[Union[HumanMessage, AIMessage]]) -> List[Dict[str, Any]]:
        if self.api_type == "chat/completions":
            result = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, SystemMessage):
                    role = "system"
                else:
                    role = "user"  # fallback
                    
                if isinstance(m.content, str):
                    content = [{"type": "text", "text": m.content}]
                else:
                    content = m.content
                    
                result.append({"role": role, "content": content})
            return result
        elif self.api_type == "responses":
            result = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, SystemMessage):
                    role = "system"
                else:
                    role = "user"  # fallback
                    
                if isinstance(m.content, str):
                    content = [{"type": "input_text", "text": m.content}]
                elif isinstance(m.content, list):
                    content = []
                    for c in m.content:
                        type_ = c.get("type", "input_text")
                        if type_ == "text":
                            content.append({
                                "type": "input_text",
                                "text": c.get("text", "")
                            })
                        elif type_ == "image_url":
                            content.append({
                                "type": "input_image",
                                "image_url": c.get("image_url", {})
                            })
                result.append({"role": role, "content": content})
            return result
                
                    