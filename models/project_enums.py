from enum import Enum

class PruningMode(Enum):
    """
    定义剪枝（pruning）的几种模式。
    - SCORE: 基于计算出的注意力分数进行剪枝。
    - RANDOM: 随机选择要保留的 token。
    - NO_PRUNING: 不进行任何剪枝，保留所有 token。
    """
    SCORE = 'score'
    RANDOM = 'random'
    NO_PRUNING = 'no_img_token_pruning'
    CDPRUNER = 'cdpruner'