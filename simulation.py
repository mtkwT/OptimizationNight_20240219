import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from loguru import logger


def log_function(
    x: float | np.ndarray,
    beta: float | np.ndarray,
) -> float | np.ndarray:
    """Log function

    Args:
        x (float | np.ndarray): x
        beta (float | np.ndarray): log関数の係数\beta

    Returns:
        float | np.ndarray: y
    """
    return beta * np.log(x + 1)


def hill_function(
    x: float | np.ndarray,
    beta: float | np.ndarray,
    alpha: float | np.ndarray,
) -> float | np.ndarray:
    """Hill function

    Args:
        x (float | np.ndarray): x
        beta (float | np.ndarray): hillのパラメータ\beta
        alpha (float | np.ndarray): hillのパラメータ\alpha

    Returns:
        float | np.ndarray: y
    """
    return beta * x / (alpha + x)


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
        """_summary_

        Args:
            a_for_beta_distribution (int | float): _description_
            b_for_beta_distribution (int | float): _description_
            seed (int): _description_

        Returns:
            pd.DataFrame: _description_
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
        not_reach_data = np.zeros((not_reach_panel_size, self.num_of_cm_logs))

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
            pd.Series: インクリメンタルなグロスリーチのリスト
        """
        array_gross_reach = (
            self.df_reach.cumsum(axis=1).sum(axis=0).to_numpy()
        ) / self.num_of_panels

        return array_gross_reach * 100

    def calculate_unique_reach(self) -> np.ndarray:
        """ユニークリーチを計算する

        Returns:
            pd.Series: インクリメンタルなユニークリーチのリスト
        """
        array_unique_reach = (
            self.df_reach.cummax(axis=1).sum(axis=0).to_numpy()
        ) / self.num_of_panels

        return array_unique_reach * 100


def main():
    # ===== リーチデータ生成 =====
    # シミュレーションのパラメータ
    num_of_panels = 1000
    num_of_cm_logs = 26
    not_reach_ratio = 0.3

    # シミュレーションの実行
    reach_data_generator = ReachDataGenerator(
        num_of_panels=num_of_panels,
        num_of_cm_logs=num_of_cm_logs,
        ratio_of_not_reach=not_reach_ratio,
    )
    df_reach = reach_data_generator.run_generate(
        a_for_beta_distribution=1,
        b_for_beta_distribution=5,
        seed=42,
    )
    # シミュレーション結果の確認
    logger.info(df_reach.head())

    # ===== リーチ計算 =====
    reach_calculator = ReachCalculator(df_reach)

    list_gross_reach = reach_calculator.calculate_gross_reach()
    logger.info(f"{list_gross_reach=}")

    list_unique_reach = reach_calculator.calculate_unique_reach()
    logger.info(f"{list_unique_reach=}")

    # ===== hill functionのフィッティング =====
    hill_nls, _ = curve_fit(
        hill_function,
        list_gross_reach,
        list_unique_reach,
        bounds=(
            (0, -np.inf),
            (100, np.inf),
        ),
    )
    logger.info(f"{hill_nls=}")

    # ===== log functionのフィッティング =====
    log_nls, _ = curve_fit(
        log_function,
        list_gross_reach,
        list_unique_reach,
    )
    logger.info(f"{log_nls=}")

    # ===== 可視化 =====
    plt.figure(figsize=(5, 3))
    x = np.linspace(0, max(list_gross_reach), 100)
    hill_y = hill_function(x, hill_nls[0], hill_nls[1])
    log_y = log_function(x, log_nls[0])
    plt.plot(x, hill_y, label="hill function", linestyle="-", alpha=0.75)
    plt.plot(x, log_y, label="log function", linestyle="--", alpha=0.75)
    plt.scatter(list_gross_reach, list_unique_reach, s=20, alpha=0.75)
    plt.xlabel("Gross reach (%)")
    plt.ylabel("Unique reach (%)")
    # y軸の範囲を0から100に設定
    # plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reach_curve.png", dpi=500)
    # シミュレーション結果の可視化
    # plt.imshow(reach_data, cmap="gray", aspect="auto")
    # plt.xlabel("CM log id")
    # plt.ylabel("Panel id")
    # plt.show()


if __name__ == "__main__":
    main()
