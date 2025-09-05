import os
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv(verbose=True)

from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from src.models.restful.chat import ChatRestfulSearch
from src.utils import Singleton
from src.logger import logger

PLACEHOLDER = "PLACEHOLDER"

class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        usage = None
        
        # Handle LLMResult
        if hasattr(response, "llm_output") and response.llm_output:
            if "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
        
        # Handle direct usage_metadata
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            cost = usage.get("cost", 0.0)
            
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            logger.info(f"| Model name: {self.model_name}. Tokens: {self.input_tokens} input tokens, {self.output_tokens} output tokens, {self.total_tokens} total tokens. Cost: ${self.total_cost:.6f}")

class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registed_models: Dict[str, Any] = {}
        self.registed_models_info: Dict[str, Any] = {}
        
    async def initialize(self, use_local_proxy: bool = False):
        self._register_openai_models(use_local_proxy=use_local_proxy)
        self._register_anthropic_models(use_local_proxy=use_local_proxy)
        self._register_google_models(use_local_proxy=use_local_proxy)
        # browser-use
        self._register_browser_models(use_local_proxy=use_local_proxy)
        
    def get_model(self, model_name: str) -> Any:
        return self.registed_models[model_name]
    
    def get_model_info(self, model_name: str) -> Any:
        return self.registed_models_info[model_name]
    
    def list_models(self) -> List[str]:
        return list(self.registed_models.keys())

    def _check_local_api_key(self, local_api_key_name: str, remote_api_key_name: str) -> str:
        api_key = os.getenv(local_api_key_name, PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(f"| Local API key {local_api_key_name} is not set, using remote API key {remote_api_key_name}")
            api_key = os.getenv(remote_api_key_name, PLACEHOLDER)
        return api_key
    
    def _check_local_api_base(self, local_api_base_name: str, remote_api_base_name: str) -> str:
        api_base = os.getenv(local_api_base_name, PLACEHOLDER)
        if api_base == PLACEHOLDER:
            logger.warning(f"| Local API base {local_api_base_name} is not set, using remote API base {remote_api_base_name}")
            api_base = os.getenv(remote_api_base_name, PLACEHOLDER)
        return api_base
    
    def _register_openai_models(self, use_local_proxy: bool = False):
        if use_local_proxy:
            logger.info("| Using local proxy for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            
            # gpt-4o
            model_name = "gpt-4o"
            model_id = "gpt-4o"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # gpt-4.1
            model_name = "gpt-4.1"
            model_id = "gpt-4.1"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # gpt-5
            model_name = "gpt-5"
            model_id = "gpt-5"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                use_responses_api=True,
                output_version="responses/v1",
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # o1
            model_name = "o1"
            model_id = "o1"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # o3
            model_name = "o3"
            model_id = "o3"

            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # gpt-4o-search-preview
            model_name = "gpt-4o-search-preview"
            model_id = "gpt-4o-search-preview"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # # deep research
            model_name = "o3-deep-research"
            model_id = "o3-deep-research"

            model = ChatRestfulSearch(
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="responses",
                model=model_id,
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # o4-mini-deep-research
            model_name = "o4-mini-deep-research"
            model_id = "o4-mini-deep-research"

            model = ChatRestfulSearch(
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_SHUBIAOBIAO_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="responses",
                model=model_id,
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
        else:
            logger.info("| Using remote API for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE")
            
            models = [
                {
                    "model_name": "gpt-4o",
                    "model_id": "gpt-4o",
                },
                {
                    "model_name": "gpt-4.1",
                    "model_id": "gpt-4.1",
                },
                {
                    "model_name": "gpt-5",
                    "model_id": "gpt-5",
                },
                {
                    "model_name": "o1",
                    "model_id": "o1",
                },
                {
                    "model_name": "o3",
                    "model_id": "o3",
                },
                {
                    "model_name": "gpt-4o-search-preview",
                    "model_id": "gpt-4o-search-preview",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                    callbacks=[TokenUsageCallbackHandler(model_name)],
                )
                self.registed_models[model_name] = model
                self.registed_models_info[model_name] = {
                    "type": "openai",
                    "model_name": model_name,
                    "model_id": model_id,
                }
                
            
    def _register_anthropic_models(self, use_local_proxy: bool = False):
        # claude37-sonnet, claude-4-sonnet
        if use_local_proxy:
            logger.info("| Using local proxy for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            
            # claude37-sonnet
            model_name = "claude-3.7-sonnet"
            model_id = "claude37-sonnet"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }

            # claude-4-sonnet
            model_name = "claude-4-sonnet"
            model_id = "claude-4-sonnet"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )   
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
        else:
            logger.info("| Using remote API for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="ANTHROPIC_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="ANTHROPIC_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE")
            
            models = [
                {
                    "model_name": "claude37-sonnet",
                    "model_id": "claude-3-7-sonnet-20250219",
                },
                {
                    "model_name": "claude-4-sonnet",
                    "model_id": "claude-4-sonnet",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = ChatAnthropic(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                    callbacks=[TokenUsageCallbackHandler(model_name)],
                )
                self.registed_models[model_name] = model
                self.registed_models_info[model_name] = {
                    "type": "anthropic",
                    "model_name": model_name,
                    "model_id": model_id,
                }
            
    def _register_google_models(self, use_local_proxy: bool = False):
        # gemini-2.5-pro
        if use_local_proxy:
            logger.info("| Using local proxy for Google models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            
            # gemini-2.5-pro
            model_name = "gemini-2.5-pro"
            model_id = "gemini-2.5-pro"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="GOOGLE_API_BASE"),
                callbacks=[TokenUsageCallbackHandler(model_name)],
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "google",
                "model_name": model_name,
                "model_id": model_id,
            }
            
        else:
            logger.info("| Using remote API for Google models")
            api_key = self._check_local_api_key(local_api_key_name="GOOGLE_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="GOOGLE_API_BASE", 
                                                    remote_api_base_name="GOOGLE_API_BASE")
            
            models = [
                {
                    "model_name": "gemini-2.5-pro",
                    "model_id": "gemini-2.5-pro",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = ChatGoogleGenerativeAI(
                    model=model_id,
                    api_key=api_key,
                    callbacks=[TokenUsageCallbackHandler(model_name)],
                )
                self.registed_models[model_name] = model
                self.registed_models_info[model_name] = {
                    "type": "google",
                    "model_name": model_name,
                    "model_id": model_id,
                }
                
    def _register_browser_models(self, use_local_proxy: bool = False):
        # browser-use
        from browser_use import ChatOpenAI
        from browser_use import ChatAnthropic
        
        if use_local_proxy:
            logger.info("| Using local proxy for Browser models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            
            # gpt-4.1
            model_name = "bs-gpt-4.1"
            model_id = "gpt-4.1"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # gpt-5
            model_name = "bs-gpt-5"
            model_id = "gpt-5"
            model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "openai",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # claude-3.7-sonnet
            model_name = "bs-claude-3.7-sonnet"
            model_id = "claude37-sonnet"
            model = ChatAnthropic(
                model=model_id,
                api_key=api_key,
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "anthropic",
                "model_name": model_name,
                "model_id": model_id,
            }
            
            # claude-4-sonnet
            model_name = "bs-claude-4-sonnet"
            model_id = "claude-4-sonnet"
            model = ChatAnthropic(
                model=model_id,
                api_key=api_key,
            )
            self.registed_models[model_name] = model
            self.registed_models_info[model_name] = {
                "type": "anthropic",
                "model_name": model_name,
                "model_id": model_id,
            }
        else:
            logger.info("| Using remote API for Browser models")
            
            # OpenAI
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE")
            
            models = [
                {
                    "model_name": "bs-gpt-4.1",
                    "model_id": "gpt-4.1",
                },
                {
                    "model_name": "bs-gpt-5",
                    "model_id": "gpt-5",
                },
            ]
                
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
                self.registed_models_info[model_name] = {
                    "type": "openai",
                    "model_name": model_name,
                    "model_id": model_id,
                }
                
            # Anthropic
            api_base = self._check_local_api_base(local_api_base_name="ANTHROPIC_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE")
            
            models = [
                {
                    "model_name": "bs-claude-3.7-sonnet",
                    "model_id": "claude37-sonnet",
                },
                {
                    "model_name": "bs-claude-4-sonnet",
                    "model_id": "claude-4-sonnet",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = ChatAnthropic(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
                self.registed_models_info[model_name] = {
                    "type": "anthropic",
                    "model_name": model_name,
                    "model_id": model_id,
                }
                
    async def init_models(self, use_local_proxy: bool = False):
        await self.initialize(use_local_proxy=use_local_proxy)
            
model_manager = ModelManager()