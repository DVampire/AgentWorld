from langchain_anthropic import ChatAnthropic

class LangchainAnthropicModel(ChatAnthropic):
    def __init__(self,
                 model: str,
                 api_key: str,
                 base_url: str,
                 **kwargs):
        super().__init__(model=model,
                         api_key=api_key,
                         base_url=base_url,
                         **kwargs)
        
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)
    
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)