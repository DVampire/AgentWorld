# Install pyqlib using 'pip install pyqlib' before running this code.
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from qlibentry import QlibEntryService

if __name__ == "__main__":
    QlibEntryService.initilization(data_provider_uri="./datasets/qlib_data/cn_data")
    FACTORS = [["2 * ((EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9))","my_factor_1"],["Std($close, 10)/$close","my_factor_2"]]
    LABEL_EXP = ["Ref($close, -2)/Ref($close, -1) - 1","my_label"]
    instruments =  "csi500"
    benchmark = "SH000905"
    start_time = "2024-07-01"
    end_time = "2024-10-01"
    for i in range(len(FACTORS)):
        RESULTS = QlibEntryService.backtest(FACTORS[i],LABEL_EXP,instruments,benchmark,start_time,end_time)
        print(RESULTS)