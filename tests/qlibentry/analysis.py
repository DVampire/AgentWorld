from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
import pandas as pd 

class Analysis():
    def __init__(self,recorder,dataset):
        self.dataset = dataset
        self.recorder = recorder
        self.pred_df = self.recorder.load_object("pred.pkl")
        self.report_normal_df = self.recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        self.positions = self.recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        self.analysis_df = self.recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        self.ic = self.recorder.load_object("sig_analysis/ic.pkl")
        self.ric = self.recorder.load_object("sig_analysis/ric.pkl")
        self.label_df = self.dataset.prepare("test", col_set="label")

        pred_df_dates = self.pred_df.index.get_level_values(level='datetime')
        self.features_df = D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'], pred_df_dates.min(), pred_df_dates.max())
        self.features_df.columns = ['label']
        self.label_df.columns = ["label"]
        self.pred_label = pd.concat([self.label_df, self.pred_df], axis=1, sort=True).reindex(self.label_df.index)

    def return_report_graph(self):
        return analysis_position.report_graph(self.report_normal_df,show_notebook=False)
    
    def return_risk_analysis_graph(self):
        return analysis_position.risk_analysis_graph(self.analysis_df,self.report_normal_df,show_notebook=False)
    
    def return_rank_label_graph(self):
        return analysis_position.rank_label_graph(self.positions,self.features_df,show_notebook=False)
    
    def return_cumulative_return_graph(self):
        return analysis_position.cumulative_return_graph(position=self.positions, report_normal=self.report_normal_df,label_data=self.features_df,show_notebook=False)
    
    def return_model_performance_graph(self):
        return analysis_model.model_performance_graph(self.pred_label,show_notebook=False)
    
    def return_score_ic_graph(self):
        return analysis_position.score_ic_graph(self.pred_label,show_notebook=False)
    