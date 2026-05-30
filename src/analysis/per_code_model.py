from typing import List

import numpy as np

from __future__ import annotations
from src.utils import PydanticBaseModel
from pathlib import Path

class PerCodeHorizonPayLoad(PydanticBaseModel):
    """ 
    数据集，每个code 对每个分片的各种主要数据统计， 方便生成各分项统计卡片。
    
    """
    # demo
    # best_code 
    # selector with one-step
    # selector with step-by-step
    # selector and tuningwith step-by-step
    type: str 
    code_label: int  # code id
    # 当前code 命中了哪些样本
    sample_ids: tuple[int, ...]  # sample id
    # 当前code 命中了样本的哪些动作
    actions: tuple[float, ...]  # action
    # 当前code 命中了样本的哪些奖励
    rewards: tuple[float, ...]  # reward
    #每个样本的形态
    morphology: tuple[str, ...]
    #每个样本的 motif
    motif: tuple[list[str], ...]
    

class PerCodeProfit(PydanticBaseModel):    
    """
    code 获得收益, (gain, loss, profit) 
    ``gain``: ``[x]``
    ``loss``: ``[x]``
    ``profit``: ``[x]``
    len(x) 是 code 数量 
    """
    gain: int
    loss: int
    profit: int 

class PerCodeDistribution(PydanticBaseModel):   
    # code sample分布, 百分比分布。
    distribution: int
    distribution_sample_count: int    
    
class MorphologyPair(PydanticBaseModel):   
    # 所有样本的所有形态
    morphology: tuple[str, ...]  # morphology string
    # 所有样本的 motif
    motif: tuple[list[str], ...]  # motif string list
    # 所有所有 morphology_motif_pair    ,key =  morphology+"_"+motifs value is profit
    morphology_motif_pair_value: dict[str, int] 

class PerCodeDiagnostic(PydanticBaseModel):
    """单个 code 的 report 级诊断数据。

    功能说明:
        汇总 code support、occupancy、dominant morphology/motif/pair 和 decoded
        profitability 等信息。

    使用场景:
        直接供 report 渲染 code-level 表格，也可用于定位弱 code、坏 code 或重复 code。
    """

    # split 中分配到该 code 的样本数量。
    distribution_sample_count: int

    # 该 code 的样本占比，等于 support / N　＊１００。
    distribution: int

    # 该 code 内占比最高的市场形态；不可计算时为 None。
    dominant_morphology: str 

    # dominant morphology 在该 code 内的占比　；不可计算时为 None。
    dominant_morphology_ratio: int

    # dominant morphology 相对全体验证集分布的 lift；不可计算时为 None。
    morphology_lift:int

    # 该 code 内占比最高的交易 motif；不可计算时为 None。
    dominant_motif: str 

    # dominant motif 在该 code 内的占比；不可计算时为 None。
    dominant_motif_ratio: int

    # 该 code 内占比最高的 morphology-motif 组合；不可计算时为 None。
    dominant_pair: str

    # dominant pair 在该 code 内的占比；不可计算时为 None。
    dominant_pair_ratio: int

    # 该 code 的 decoded mean advantage vs flat；不可计算时为 None。
    decoded_mean_advantage: int

    # 该 code 的 decoded win rate vs flat；不可计算时为 None。
    decoded_win_rate: int

    # 该 code 的手续费拖累比例；不可计算时为 None。
    fee_drag: int


class OverallCodeHorizonPayLoad(PydanticBaseModel):     
    
 
    #code分布。 形状是code数量
    demo_per_code_distribution: List[PerCodeDistribution] 
    #code分布。 形状是code数量
    selector_per_code_distribution: List[PerCodeDistribution]   
    #code分布。 形状是code数量
    best_code_per_code_distribution: List[PerCodeDistribution]     
    #---    
    #累积利润序列, shape len(sample_ids)
    dp_profit_sequence: tuple[int, ...]
    demo_profit_sequence: tuple[int, ...]
    best_code_profit_sequence: tuple[int, ...]
    #code分布。 形状是code数量
    selector_one_step_profit_sequence: tuple[int, ...]
    selector_step_by_step_profit_sequence: tuple[int, ...]          
    selector_tuning_step_by_step_profit_sequence: tuple[int, ...]
    #一直做多hold
    base_line_profit_sequence: tuple[int, ...]
    
    #---
    #code 获得收益, shape code数量
    demo_per_code_profit: List[PerCodeProfit]    
    best_code_per_code_profit: List[PerCodeProfit]     
    selector_one_step_per_code_profit: List[PerCodeProfit]  
    selector_step_by_step_per_code_profit: List[PerCodeProfit] 
    selector_tuning_step_by_step_per_code_profit: List[PerCodeProfit] 
    #--- 行情和动作模式的分布
    dp_morphology: MorphologyPair
    demo_morphology: MorphologyPair
    best_code_morphology: MorphologyPair
    selector_one_step_morphology: MorphologyPair
    selector_step_by_step_morphology: MorphologyPair
    selector_tuning_step_by_step_morphology: MorphologyPair
    
    
    #code诊断。 形状是code数量
    demo_per_code_diagnostic: List[PerCodeDiagnostic] 
    #code诊断。 形状是code数量
    selector_per_code_diagnostic: List[PerCodeDiagnostic] 
    #code诊断。 形状是code数量
    best_code_per_code_diagnostic: List[PerCodeDiagnostic]     
    #code诊断。 形状是code数量
    selector_one_step_per_code_diagnostic: List[PerCodeDiagnostic]  
    #code诊断。 形状是code数量
    selector_step_by_step_per_code_diagnostic: List[PerCodeDiagnostic] 
    #code诊断。 形状是code数量
    selector_tuning_step_by_step_per_code_diagnostic: List[PerCodeDiagnostic] 
    
    # 参与 code distribution 统计的样本数。
    code_distribution_total_sample_count: int