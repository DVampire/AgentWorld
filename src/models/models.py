import os
from openai import AsyncOpenAI
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.models.langchain_openai import LangchainOpenAIModel
from src.models.langchain_anthropic import LangchainAnthropicModel
from src.models.langchain_google import LangchainGoogleModel
from src.models.restful import RestfulModel, RestfulSearchModel
from src.utils import Singleton

PLACEHOLDER = "PLACEHOLDER"


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registed_models: Dict[str, Any] = {}
        
    def init_models(self, use_local_proxy: bool = False):
        self._register_openai_models(use_local_proxy=use_local_proxy)
        self._register_anthropic_models(use_local_proxy=use_local_proxy)
        self._register_google_models(use_local_proxy=use_local_proxy)
        
    def get_model(self, model_name: str) -> Any:
        return self.registed_models[model_name]
    
    def list_models(self) -> List[str]:
        return list(self.registed_models.keys())

    def _check_local_api_key(self, local_api_key_name: str, remote_api_key_name: str) -> str:
        api_key = os.getenv(local_api_key_name, PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(f"Local API key {local_api_key_name} is not set, using remote API key {remote_api_key_name}")
            api_key = os.getenv(remote_api_key_name, PLACEHOLDER)
        return api_key
    
    def _check_local_api_base(self, local_api_base_name: str, remote_api_base_name: str) -> str:
        api_base = os.getenv(local_api_base_name, PLACEHOLDER)
        if api_base == PLACEHOLDER:
            logger.warning(f"Local API base {local_api_base_name} is not set, using remote API base {remote_api_base_name}")
            api_base = os.getenv(remote_api_base_name, PLACEHOLDER)
        return api_base
    
    def _register_openai_models(self, use_local_proxy: bool = False):
        if use_local_proxy:
            logger.info("Using local proxy for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            
            # gpt-4o
            model_name = "gpt-4o"
            model_id = "gpt-4o"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            
            # gpt-4.1
            model_name = "gpt-4.1"
            model_id = "gpt-4.1"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            
            # gpt-5
            model_name = "gpt-5"
            model_id = "gpt-5"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                use_responses_api=True,
                output_version="responses/v1"
            )
            self.registed_models[model_name] = model

            
            # o1
            model_name = "o1"
            model_id = "o1"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model

            # o3
            model_name = "o3"
            model_id = "o3"

            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            
            # gpt-4o-search-preview
            model_name = "gpt-4o-search-preview"
            model_id = "gpt-4o-search-preview"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
            )
            self.registed_models[model_name] = model
            
            # # deep research
            model_name = "o3-deep-research"
            model_id = "o3-deep-research"

            model = RestfulSearchModel(
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="responses",
                model=model_id,
            )
            self.registed_models[model_name] = model
            
            # o4-mini-deep-research
            model_name = "o4-mini-deep-research"
            model_id = "o4-mini-deep-research"

            model = RestfulSearchModel(
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_SHUBIAOBIAO_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="responses",
                model=model_id,
            )
            self.registed_models[model_name] = model
            
        else:
            logger.info("Using remote API for OpenAI models")
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
                model = LangchainOpenAIModel(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
                
            
    def _register_anthropic_models(self, use_local_proxy: bool = False):
        # claude37-sonnet, claude-4-sonnet
        if use_local_proxy:
            logger.info("Using local proxy for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            
            # claude37-sonnet
            model_name = "claude-3.7-sonnet"
            model_id = "claude37-sonnet"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
            )
            self.registed_models[model_name] = model

            # claude-4-sonnet
            model_name = "claude-4-sonnet"
            model_id = "claude-4-sonnet"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
            )
            self.registed_models[model_name] = model

        else:
            logger.info("Using remote API for Anthropic models")
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
                model = LangchainAnthropicModel(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
            
    def _register_google_models(self, use_local_proxy: bool = False):
        # gemini-2.5-pro
        if use_local_proxy:
            logger.info("Using local proxy for Google models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            
            # gemini-2.5-pro
            model_name = "gemini-2.5-pro"
            model_id = "gemini-2.5-pro"
            model = LangchainOpenAIModel(
                model=model_id,
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="GOOGLE_API_BASE"),
            )
            self.registed_models[model_name] = model
            
        else:
            logger.info("Using remote API for Google models")
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
                model = LangchainGoogleModel(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
            
            
model_manager = ModelManager()