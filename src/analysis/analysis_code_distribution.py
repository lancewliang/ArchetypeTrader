
from src.utils import PydanticBaseModel


class VQCodeDistributionPayload(PydanticBaseModel):
    """ 
    validation split 的
    """

    # code 使用分布，索引为 code id，值为 occupancy probability。
    # validation split /code_distribution_total_sample_count
    code_distribution: tuple[float, ...]
    
     # code 使用分布，索引为 code id，值为 assigned 样本数。
    code_distribution_sample_count: tuple[int, ...]
    
    # validation split 中达到 active occupancy 阈值的 code id。
    active_codes: tuple[int, ...] 
    
    # 参与 code distribution 统计的样本数。
    code_distribution_total_sample_count: int