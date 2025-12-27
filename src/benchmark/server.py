from src.benchmark.types import Benchmark

"aime24"

class BenchmarkManager(Benchmark):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    async def initialize(self):
        pass
        
    async def get_item(self, index: int):
        pass
    
    async def answer_item(self, item: dict):
        pass
    
    async def evaluate(self):
        pass

benchmark_manager = BenchmarkManager()