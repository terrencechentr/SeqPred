# 导入所有模型配置和类
from models.csept import CseptConfig, CseptModel, CspetForSequencePrediction, CseptOutput
from models.csept_smooth import CseptSmoothConfig, CseptSmoothModel, CspetSmoothForSequencePrediction, CseptSmoothOutput
from models.csept_smooth_improved import CspetSmoothWithDirectionLoss, CspetSmoothMultiTask
from models.sept import SeptConfig, Qwen2ForSept, SeptOutput

# 注册所有配置到AutoConfig
try:
    from transformers import AutoConfig, AutoModel
    
    # 注册配置
    AutoConfig.register("csept", CseptConfig)
    AutoConfig.register("csept_smooth", CseptSmoothConfig)
    AutoConfig.register("sept", SeptConfig)
    
    # 注册模型
    AutoModel.register(CseptConfig, CspetForSequencePrediction)
    AutoModel.register(CseptSmoothConfig, CspetSmoothForSequencePrediction)
    AutoModel.register(SeptConfig, Qwen2ForSept)
    
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
]

