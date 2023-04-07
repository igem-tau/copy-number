import numpy as np
import pandas as pd
from Bio import motifs


def split_by_buckets(df: pd.DataFrame, num_buckets: int, split_column: str):
    """

    :param df:
    :param num_buckets:
    :param split_column:
    :return:
    """
    df_sorted = df.sort_values(split_column)
    buckets = np.array_split(df_sorted, num_buckets)
    return buckets


def get_pssm_using_motifs_for_buckets(buckets: list, show=False):
    buckets_with_motifs = {i: motifs.create(b["Promoter Sequence"]) for i, b in enumerate(buckets)}
    if show:
        for b in buckets_with_motifs:
            # lf = LogoFormat(buckets_with_motifs[b], format="png")
            # lg = LogoGenerator()
            # fig = lg.create_logo_figure(lf)
            # plt.show(fig)
            # Todo: need to fix: strange, it keeps generating html and not jpeg
            buckets_with_motifs[b].weblogo(f"motifs_b{b}.jpeg", first_index=START_INDEX, format="jpeg")
            # img = mpimg.imread(f"motifs_b{b}.png")
            # plt.imshow(img)
            # plt.show()

    buckets_with_pssm = {i: pd.DataFrame(m.counts) for i, m in buckets_with_motifs.items()}
    return buckets_with_pssm



