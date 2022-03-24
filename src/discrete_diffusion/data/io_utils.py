import math
from typing import List, Tuple

import numpy as np


def random_split_sequence(sequence: List, split_ratio: float) -> Tuple[List, List]:
    f"""
    Splits a sequence randomly into two sequences, the first having {split_ratio}% of the elements
    and the second having {1-split_ratio}%.
    :param sequence: sequence to be split.
    :param split_ratio: percentage of the elements falling in the first sequence.
    :return: subseq_1, subseq_2
    """

    idxs = np.arange(len(sequence))
    np.random.shuffle(idxs)

    support_upperbound = math.ceil(split_ratio * len(sequence))
    split_sequence_1_idxs = idxs[:support_upperbound]
    split_sequence_2_idxs = idxs[support_upperbound:]

    split_seq_1 = [sequence[idx] for idx in split_sequence_1_idxs]
    split_seq_2 = [sequence[idx] for idx in split_sequence_2_idxs]

    return split_seq_1, split_seq_2
