import pandas as pd
import numpy as np


class ReachDataGenerator:
    """グロスリーチとユニークリーチのデータを生成する

    データの形式は以下のようになる
    panel_id, cm_log_id_1, cm_log_id_2, cm_log_id_3,  ...
    00001.         1.              0.              1. ...
    00002.         0.              1.              0. ...
    00003.         1.              0.              1. ...
    00004.         1.              1.              0. ...
    00005.         0.              0.              1. ...
    00006.         0.              0.              0. ...
    ...

    パネルの数と、CMの放送回数（ = cm_log_{id}のidの数）を指定することができる
    """

    def __init__(
        self,
        num_of_panels: int,
        num_of_cm_logs: int,
        ratio_of_not_reach: float,
    ) -> None:
        """

        Args:
            num_of_panels (int): パネル数
            num_of_cm_logs (int): CM放送回数
            ratio_of_not_reach (float): 確実にリーチしない割合
        """
        self.num_of_panels = num_of_panels
        self.num_of_cm_logs = num_of_cm_logs
        self.ratio_of_not_reach = ratio_of_not_reach

    def run_generate(
        self,
        a_for_beta_distribution: int | float,
        b_for_beta_distribution: int | float,
        seed: int,
    ) -> pd.DataFrame:
        """リーチデータの生成を実行する
        パネルごとにCMへの接触確率は異なると仮定し、接触確率はベータ分布からドローする
        self.num_of_panels * (1 - self.ratio_of_not_reach)人のパネルに対してのみ、
        リーチが発生すると仮定し、ベータ分布から接触確率をドローし、実際のリーチデータを二項分布からドローする（ベータ二項分布）

        Args:
            a_for_beta_distribution (int | float): ベータ分布のパラメータa
            b_for_beta_distribution (int | float): ベータ分布のパラメータb
            seed (int): サンプリングを固定させるための乱数シード

        Returns:
            pd.DataFrame: パネル数xCM放送回数のリーチデータ
        """

        panel_ids = [f"{str(i).zfill(5)}." for i in range(1, self.num_of_panels + 1)]
        cm_log_ids = [f"cm_log_id_{i}" for i in range(1, self.num_of_cm_logs + 1)]

        # パネルごとのリーチ確率を生成
        np.random.seed(seed)
        # リーチが発生しうる 100 * (1 - ratio_of_not_reach)%のパネルに対して確率をベータ分布から生成
        reach_panel_size = int(self.num_of_panels * (1 - self.ratio_of_not_reach))
        panel_reach_probabilities = np.random.beta(
            a=a_for_beta_distribution,
            b=b_for_beta_distribution,
            size=reach_panel_size,
        )
        # パネルごとのリーチデータ(0,1)を生成
        for i, p in enumerate(panel_reach_probabilities):
            panel_reach_data = np.random.binomial(n=1, p=p, size=self.num_of_cm_logs)
            if i == 0:
                reach_data = panel_reach_data
                continue
            reach_data = np.vstack((reach_data, panel_reach_data))

        # リーチが発生しないパネルに対しては、全てのCMに対してリーチが発生しないとする
        not_reach_panel_size = self.num_of_panels - reach_panel_size
        not_reach_data = np.zeros(shape=(not_reach_panel_size, self.num_of_cm_logs))

        return pd.DataFrame(
            data=np.vstack((reach_data, not_reach_data)),
            index=panel_ids,
            columns=cm_log_ids,
        )


class ReachCalculator:
    """ReachDataGeneratorで生成したリーチデータを用いて、グロスリーチとユニークリーチを計算する
    データの形式は以下のようになる
    panel_id, cm_log_id_1, cm_log_id_2, cm_log_id_3,  ...
    00001.         1.              0.              1. ...
    00002.         0.              1.              0. ...
    00003.         1.              0.              1. ...
    00004.         1.              1.              0. ...
    00005.         0.              0.              1. ...
    00006.         0.              0.              0. ...

    - グロスリーチは全ての1の数を足し合わせたもの
    - ユニークリーチはpanel_idごとに、1が一度でも出たら1として、その数を足し合わせたもの
    """

    def __init__(
        self,
        df_reach: pd.DataFrame,
    ) -> None:
        self.df_reach = df_reach
        self.num_of_panels = df_reach.shape[0]

    def calculate_gross_reach(self) -> np.ndarray:
        """グロスリーチを計算する

        Returns:
            pd.Series: インクリメンタルなグロスリーチのnumpy配列
        """
        array_gross_reach = (
            self.df_reach.cumsum(axis=1).sum(axis=0).to_numpy()
        ) / self.num_of_panels

        return array_gross_reach * 100

    def calculate_unique_reach(self) -> np.ndarray:
        """ユニークリーチを計算する

        Returns:
            pd.Series: インクリメンタルなユニークリーチのnumpy配列
        """
        array_unique_reach = (
            self.df_reach.cummax(axis=1).sum(axis=0).to_numpy()
        ) / self.num_of_panels

        return array_unique_reach * 100
