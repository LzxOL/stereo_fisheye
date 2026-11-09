"""双目鱼眼 pipeline：球面扫描、初始索引和 Context / GRU 迭代优化。"""

from .network import StereoBaseline

__all__ = ["StereoBaseline"]
