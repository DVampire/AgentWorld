import sys
from pathlib import Path
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset import DatasetH,DataHandler

import pandas as pd
from .analysis import Analysis

class SingleFactorModel:
    def fit(self,dataset,**kwargs):
        pass

    def predict(self, dataset,segment="test"):
        df = dataset.prepare(segment, col_set=["feature", "label"])
        score = df["feature"].iloc[:, 0].rename("score")
        return score

class QlibEntryService:
    ## Using singletion pattern
    _initilizaed = False
    _provider_uri = None

    @staticmethod
    def initilization(data_provider_uri: str):
        if not QlibEntryService._initilizaed:
            if not exists_qlib_data(data_provider_uri):
                raise ValueError(f"Qlib data is not found in {data_provider_uri}")
            qlib.init(provider_uri=data_provider_uri, 
                      region=REG_CN
)
            QlibEntryService._initilizaed = True
            QlibEntryService._provider_uri = data_provider_uri
        else:
            if data_provider_uri != QlibEntryService._provider_uri:
                raise RuntimeError(f"Qlib has been initialized with provider_uri {QlibEntryService._provider_uri}, cannot re-initialize with different provider_uri {data_provider_uri}")
            
    @staticmethod
    def generate_signal(FACTOR:list[str,str],LABEL_EXP:list[str,str],instruments:str,start_time:str,end_time:str)->pd.DataFrame:
        if QlibEntryService._initilizaed is False:
            raise RuntimeError("QlibEntryService has not been initialized, please call initilization(provider_uri) first.")
        fields = [FACTOR[0]]
        factor_name = [FACTOR[1]]
        labels_fields = [LABEL_EXP[0]]
        labels_names  = [LABEL_EXP[1]]
        data_loader_config = {
            "feature":(fields, factor_name),
            "label":(labels_fields, labels_names),
        }
        data_loader = QlibDataLoader(config=data_loader_config)


        handler = DataHandler(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader)
        

        dataset = DatasetH(handler=handler, segments={"test": (start_time, end_time)})
        print(dataset.prepare("test", col_set=["feature","label"]).head())
        return dataset
    
    @staticmethod
    def workflow_backtest(FACTOR:list[str,str],LABEL_EXP:list[str,str],backtestdir:str,instruments:str,benchmark:str,start_time:str,end_time:str):
        dataset = QlibEntryService.generate_signal(FACTOR,LABEL_EXP,instruments,start_time,end_time)
        model = SingleFactorModel()
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "model": model,
                    "dataset": dataset,
                    "signal": "score",
                    "topk": 50,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": start_time,
                "end_time": end_time,
                "account": 100000000,
                "benchmark": benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        # backtest and analysis
        with R.start(experiment_name="backtest_analysis",uri=backtestdir):
            recorder = R.get_recorder()
            
            # prediction
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()
            # backtest & analysis
            sar = SigAnaRecord(recorder)
            sar.generate()
            par = PortAnaRecord(recorder, port_analysis_config, "day")
            par.generate()

        analysis = Analysis(recorder, dataset)
        return analysis
    
    @staticmethod
    def backtest(FACTOR:list[str,str],LABEL_EXP:list[str,str],instruments:str,benchmark:str,start_time:str,end_time:str,backtestdir:str="backtest",result_dir:str="results"):
        ANALYSIS = QlibEntryService.workflow_backtest(FACTOR,LABEL_EXP,backtestdir,instruments,benchmark,start_time,end_time)
        # images_dict = {}
        
        # images_dict["report_graph"] = ANALYSIS.return_report_graph()
        # images_dict["risk_analysis_graph"] = ANALYSIS.return_risk_analysis_graph()
        # images_dict["rank_label_graph"] = ANALYSIS.return_rank_label_graph()
        # images_dict["cumulative_return_graph"] = ANALYSIS.return_cumulative_return_graph()
        # images_dict["model_performance_graph"] = ANALYSIS.return_model_performance_graph()
        # images_dict["score_ic_graph"] = ANALYSIS.return_score_ic_graph()
        # images_path_list = []
        ## Save images

        # for idx,(key,fig) in enumerate(images_dict.items()):
        #     for idx2,image in enumerate(fig):
        #         path = Path(rf"./{backtestdir}/{result_dir}/figure_{idx}_{key}_{idx2}_{FACTOR[1]}.png")
        #         if path.parent.exists() is False:
        #             path.parent.mkdir(parents=True, exist_ok=True)
        #         image.write_image(path)
        #         images_path_list.append(path)

        backtest_results = {
            "excess_return":ANALYSIS.analysis_df.to_dict(),
            "costs":ANALYSIS.report_normal_df["total_cost"].sum(),
            "ic":ANALYSIS.ic,
            "ric":ANALYSIS.ric
        }

        return backtest_results 

    # @staticmethod
    # def generate_signal_quickly(FACTOR:str,LABEL_EXP:str,instruments:str,start_time:str,end_time:str)->pd.DataFrame:
    #     if QlibEntryService._initilizaed is False:
    #         raise RuntimeError("QlibEntryService has not been initialized, please call initilization(provider_uri) first.")
    #     fields = [FACTOR[0]]
    #     factor_name = [FACTOR[1]]
    #     labels_fields = [LABEL_EXP[0]]
    #     labels_names  = [LABEL_EXP[1]]
    #     data_loader_config = {
    #         "feature":(fields, factor_name),
    #         "label":(labels_fields, labels_names),
    #     }
    #     data_loader = QlibDataLoader(config=data_loader_config)
    #     df = data_loader.load(instruments=instruments, start_time=start_time, end_time=end_time, col_set=["feature", "label"])
    #     df['score'] = df[('feature',factor_name[0])]
    #     return df['score']

    # @staticmethod
    # def run_backtest(FACTOR:list[str,str],LABEL_EXP:list[str,str],instruments:str,benchmark:str,start_time:str,end_time:str):
    #     df = QlibEntryService.generate_signal_quickly(FACTOR,LABEL_EXP,instruments,start_time,end_time)
    #     STRATEGY_CONFIG = {
    #         "topk": 50,
    #         "n_drop": 5,
    #         # pred_score, pd.Series
    #         "signal": df,
    #             }
    #     strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    #     report_normal, positions_normal = backtest_daily(
    #         start_time=start_time, end_time=end_time, strategy=strategy_obj
    #     )
    #     analysis = dict()
    #     # default frequency will be daily (i.e. "day")
    #     analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    #     analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    #     analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    #     pprint(analysis_df)
    