from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='../../checkpoints/MiniCPM-V-2'),
    'MiniCPMV-Sensitive_fiu': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiu'),
    'MiniCPMV-Sensitive_fiofp': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp'),
    'MiniCPMV-Sensitive_fiofp_finetune': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_finetune'),
    'MiniCPMV-Sensitive_fiofp_multidelete': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_multidelete'),
    'MiniCPMV-Sensitive_fiofp_llmu': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_llmu'),
    'MiniCPMV-Sensitive_fiofp_kul': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_kul'),
    'MiniCPMV-Sensitive_fiofp_salun': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_salun'),
    'MiniCPMV-Sensitive_fiofp_emmu': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_emmu'),
    'MiniCPMV-Sensitive_fiofp_emmu_hard': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_emmu_hard'),
    'MiniCPMV-Sensitive_fiofp_emmu_hard_one_negative': partial(MiniCPM_V, model_path='../../output/minicpmv_sensitive_fiofp_10_emmu_hard_one_negative'),
}

supported_VLM = {}

model_groups = [
    minicpm_series
]

for grp in model_groups:
    supported_VLM.update(grp)
