"""Qlib backtesting environment implementation."""


from typing import Any, Dict, List, Union, Optional, Any, Dict, Type
from pydantic import BaseModel, Field, ConfigDict
from src.logger import logger
from src.environment.server import ecp
from src.environment.types import Environment
from src.registry import ENVIRONMENT

# from src.test.qlibentry import QlibEntryService


## Some functon hasn't been implemented yet. Refer to https://qlib.readthedocs.io/en/latest/reference/api.html#module-qlib.data.base and click operator to see all supported operations.
_EXPRESSION_RULES = """  Only the following operations are allowed in expressions: 
  ### **Cross-sectional Functions**
  - **RANK(A)**: Ranking of each element in the cross-sectional dimension of A.
  - **ZSCORE(A)**: Z-score of each element in the cross-sectional dimension of A.
  - **MEAN(A)**: Mean value of each element in the cross-sectional dimension of A.
  - **STD(A)**: Standard deviation in the cross-sectional dimension of A.
  - **SKEW(A)**: Skewness in the cross-sectional dimension of A.
  - **KURT(A)**: Kurtosis in the cross-sectional dimension of A.
  - **MAX(A)**: Maximum value in the cross-sectional dimension of A.
  - **MIN(A)**: Minimum value in the cross-sectional dimension of A.
  - **MEDIAN(A)**: Median value in the cross-sectional dimension of A

  ### **Time-Series Functions**
  - **DELTA(A, n)**: Change in value of A over n periods.
  - **DELAY(A, n)**: Value of A delayed by n periods.
  - **TS_MEAN(A, n)**: Mean value of sequence A over the past n days.
  - **TS_SUM(A, n)**: Sum of sequence A over the past n days.
  - **TS_RANK(A, n)**: Time-series rank of the last value of A in the past n days.
  - **TS_ZSCORE(A, n)**: Z-score for each sequence in A over the past n days.
  - **TS_MEDIAN(A, n)**: Median value of sequence A over the past n days.
  - **TS_PCTCHANGE(A, p)**: Percentage change in the value of sequence A over p periods.
  - **TS_MIN(A, n)**: Minimum value of A in the past n days.
  - **TS_MAX(A, n)**: Maximum value of A in the past n days.
  - **TS_ARGMAX(A, n)**: The index (relative to the current time) of the maximum value of A over the past n days.
  - **TS_ARGMIN(A, n)**: The index (relative to the current time) of the minimum value of A over the past n days.
  - **TS_QUANTILE(A, p, q)**: Rolling quantile of sequence A over the past p periods, where q is the quantile value between 0 and 1.
  - **TS_STD(A, n)**: Standard deviation of sequence A over the past n days.
  - **TS_VAR(A, p)**: Rolling variance of sequence A over the past p periods.
  - **TS_CORR(A, B, n)**: Correlation coefficient between sequences A and B over the past n days.
  - **TS_COVARIANCE(A, B, n)**: Covariance between sequences A and B over the past n days.
  - **TS_MAD(A, n)**: Rolling Median Absolute Deviation of sequence A over the past n days.
  - **PERCENTILE(A, q, p)**: Quantile of sequence A, where q is the quantile value between 0 and 1. If p is provided, it calculates the rolling quantile over the past p periods.
  - **HIGHDAY(A, n)**: Number of days since the highest value of A in the past n days.
  - **LOWDAY(A, n)**: Number of days since the lowest value of A in the past n days.
  - **SUMAC(A, n)**: Cumulative sum of A over the past n days.

  ### **Moving Averages and Smoothing Functions**
  - **SMA(A, n, m)**: Simple moving average of A over n periods with modifier m.
  - **WMA(A, n)**: Weighted moving average of A over n periods, with weights decreasing from 0.9 to 0.9^(n).
  - **EMA(A, n)**: Exponential moving average of A over n periods, where the decay factor is 2/(n+1).
  - **DECAYLINEAR(A, d)**: Linearly weighted moving average of A over d periods, with weights increasing from 1 to d.

  ### **Mathematical Operations**
  - **PROD(A, n)**: Product of values in A over the past n days. Use `*` for general multiplication.
  - **LOG(A)**: Natural logarithm of each element in A.
  - **SQRT(A)**: Square root of each element in A.
  - **POW(A, n)**: Raise each element in A to the power of n.
  - **SIGN(A)**: Sign of each element in A, one of 1, 0, or -1.
  - **EXP(A)**: Exponential of each element in A.
  - **ABS(A)**: Absolute value of A.
  - **MAX(A, B)**: Maximum value between A and B.
  - **MIN(A, B)**: Minimum value between A and B.
  - **INV(A)**: Reciprocal (1/x) of each element in sequence A.
  - **FLOOR(A)**: Floor of each element in sequence A.
  
  ### **Conditional and Logical Functions**
  - **COUNT(C, n)**: Count of samples satisfying condition C in the past n periods. Here, C is a logical expression, e.g., `$close > $open`.
  - **SUMIF(A, n, C)**: Sum of A over the past n periods if condition C is met. Here, C is a logical expression.
  - **FILTER(A, C)**: Filtering multi-column sequence A based on condition C. Here, C is presented in a logical expression form, with the same size as A.
  - **(C1)&&(C2)**: Logical operation "and". Both C1 and C2 are logical expressions, such as A > B.
  - **(C1)||(C2)**: Logical operation "or". Both C1 and C2 are logical expressions, such as A > B.
  - **(C1)?(A):(B)**: Logical operation "If condition C1 holds, then A, otherwise B". C1 is a logical expression, such as A > B.

  ### **Regression and Residual Functions**
  - **SEQUENCE(n)**: A single-column sequence of length n, ranging from 1 to integer n. `SEQUENCE()` should always be nested in `REGBETA()` or `REGRESI()` as argument B.
  - **REGBETA(A, B, n)**: Regression coefficient of A on B using the past n samples, where A MUST be a multi-column sequence and B a single-column or multi-column sequence.
  - **REGRESI(A, B, n)**: Residual of regression of A on B using the past n samples, where A MUST be a multi-column sequence and B a single-column or multi-column sequence.

  ### **Technical Indicators**
  - **RSI(A, n)**: Relative Strength Index of sequence A over n periods. Measures momentum by comparing the magnitude of recent gains to recent losses.
  - **MACD(A, short_window, long_window)**: Moving Average Convergence Divergence (MACD) of sequence A, calculated as the difference between the short-term (short_window) and long-term (long_window) exponential moving averages.
  - **BB_MIDDLE(A, n)**: Middle Bollinger Band, calculated as the n-period simple moving average of sequence A.
  - **BB_UPPER(A, n)**: Upper Bollinger Band, calculated as middle band plus two standard deviations of sequence A over n periods.
  - **BB_LOWER(A, n)**: Lower Bollinger Band, calculated as middle band minus two standard deviations of sequence A over n periods.


  Note that:
  - Only the variables provided in data (e.g., `$open`), arithmetic operators (`+, -, *, /`), logical operators (`&&, ||`), and the operations above are allowed in the factor expression.
  - Make sure your factor expression contain at least one variables within the dataframe columns (e.g. $open), combined with registered operations above. Do NOT use any undeclared variable (e.g. 'n', 'w_1') and undefined symbols (e.g., '=') in the expression. 
  - Pay attention to the distinction between operations with the TS prefix (e.g., `TS_STD()`) and those without (e.g., `STD()`). 
"""



@ENVIRONMENT.register_module(force=True)
class QlibEnvironment(Environment):
    model_config = ConfigDict(arbitrary_types_allowed=True,extra="allow")
    name: str = Field(default="qlib_environment", description="The name of the Qlib environment."),
    description: str = Field(default="Qlib backtest environment for factor construction", description="The description of the Qlib environment.")
    metadata: Dict[str, Any] = Field(default = {
        "has_vision": False,
        "addtional_rules":{
            "state": _EXPRESSION_RULES,
        }
    }
    , description="The metadata for the Qlib backtestenvironment.")
    require_grad: bool = Field(default=False, description="Whether to require gradients for the environment.")    

    def __init__(
            self,
            backtestdir: str,
            universe: str = "csi500",
            bechmark: str = "SH000905",
            result_dir: str = "results",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.backtestdir = backtestdir
        self.universe = universe
        self.bechmark = bechmark
        self.backtest_results = None
        self.result_dir = result_dir

    @classmethod
    async def initilization(cls,data_provider_uri: str) -> "QlibEnvironment":
        cls.data_provider_uri = data_provider_uri
        logger.info(f"Initializing Qlib environment with data provider URI: {cls.data_provider_uri}")
        QlibEntryService.initilization(cls.data_provider_uri)
        logger.info(f"Qlib environment initialized successfully.")

    @classmethod
    def get_service(self) -> Type[QlibEntryService]:
        return QlibEntryService
    
    def backtest(self,FACTOR:list[str,str],LABEL_EXP:list[str,str],instruments:str,benchmark:str,start_time:str,end_time:str):
        return QlibEntryService.backtest(FACTOR,LABEL_EXP,instruments=instruments,benchmark=benchmark,start_time=start_time,end_time=end_time)
    
    # def mutithread_backtest(self,FACTORS:list[list[str,str]],LABEL_EXP:list[str,str],instruments:str,benchmark:str,start_time:str,end_time:str):
    #     task_list = []
    #     for factor in FACTORS:
    #         task_list.append((self.backtest,(factor,LABEL_EXP,instruments,benchmark,start_time,end_time),{}))
        
    #     results = mutliprocess(task_list, max_workers=2, timeout_s=600.0)
    #     return results
