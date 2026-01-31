from src.logger import logger

__version__ = "0.1.8"

from .variable import Variable
from .loss import TextLoss
from .model import BlackboxLLM
from .engine import EngineLM, get_engine
from .optimizer import TextualGradientDescent, TGD
from .config import set_backward_engine, SingletonBackwardEngine
from .autograd import sum, aggregate

singleton_backward_engine = SingletonBackwardEngine()