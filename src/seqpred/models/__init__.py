# 导入所有模型配置和类
from models.csept import CseptConfig, CseptModel, CspetForSequencePrediction, CseptOutput
from models.csept_smooth import CseptSmoothConfig, CseptSmoothModel, CspetSmoothForSequencePrediction, CseptSmoothOutput
try:
    from models.csept_smooth_improved import CspetSmoothWithDirectionLoss, CspetSmoothMultiTask
except ImportError:
    CspetSmoothWithDirectionLoss = None
    CspetSmoothMultiTask = None
from models.sept import SeptConfig, Qwen2ForSept, SeptOutput
from models.rnn_naive.configuration_rnn_naive import RnnNaiveConfig
from models.rnn_naive.modeling_rnn_naive import RnnNaiveModel
from models.lstm_naive.configuration_lstm_naive import LstmNaiveConfig
from models.lstm_naive.modeling_lstm_naive import LstmNaiveModel

# 注册所有配置到AutoConfig
try:
    from transformers import AutoConfig, AutoModel
    
    # 注册配置
    AutoConfig.register("csept", CseptConfig)
    AutoConfig.register("csept_smooth", CseptSmoothConfig)
    AutoConfig.register("sept", SeptConfig)
    AutoConfig.register("rnn_naive", RnnNaiveConfig)
    AutoConfig.register("lstm_naive", LstmNaiveConfig)
    
    # 注册模型
    AutoModel.register(CseptConfig, CspetForSequencePrediction)
    AutoModel.register(CseptSmoothConfig, CspetSmoothForSequencePrediction)
    AutoModel.register(SeptConfig, Qwen2ForSept)
    AutoModel.register(RnnNaiveConfig, RnnNaiveModel)
    AutoModel.register(LstmNaiveConfig, LstmNaiveModel)
    
except ImportError:
    pass

__all__ = [
    # CSEPT
    "CseptConfig",
    "CseptModel", 
    "CspetForSequencePrediction",
    "CseptOutput",
    
    # CSEPT Smooth
    "CseptSmoothConfig",
    "CseptSmoothModel",
    "CspetSmoothForSequencePrediction",
    "CseptSmoothOutput",
    
    # CSEPT Smooth Improved
    "CspetSmoothWithDirectionLoss",
    "CspetSmoothMultiTask",
    
    # Sept
    "SeptConfig",
    "Qwen2ForSept",
    "SeptOutput",
    
    # RNN Naive
    "RnnNaiveConfig",
    "RnnNaiveModel",
    
    # LSTM Naive
    "LstmNaiveConfig",
    "LstmNaiveModel",
]

