import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from loguru import logger

from src.curve_function import hill_function, log_function
from src.simulator import ReachDataGenerator, ReachCalculator

# シミュレーションのパラメータ
NUM_OF_PANELS = 1000
NUM_OF_CM_LOGS = 26
NOT_REACH_RATIO = 0.3

A_FOR_BETA_DISTRIBUTION = 1
B_FOR_BETA_DISTRIBUTION = 5
SEED = 42


def run_generate_reach_data(
    num_of_panels: int,
    num_of_cm_logs: int,
    not_reach_ratio: float,
    a_for_beta_distribution: float,
    b_for_beta_distribution: float,
    seed: int,
) -> pd.DataFrame:
    reach_data_generator = ReachDataGenerator(
        num_of_panels=num_of_panels,
        num_of_cm_logs=num_of_cm_logs,
        ratio_of_not_reach=not_reach_ratio,
    )
    df_reach = reach_data_generator.run_generate(
        a_for_beta_distribution=a_for_beta_distribution,
        b_for_beta_distribution=b_for_beta_distribution,
        seed=seed,
    )
    return df_reach


def run_calculate_reach(
    df_reach: pd.DataFrame,
) -> (np.ndarray, np.ndarray):
    reach_calculator = ReachCalculator(df_reach)

    array_gross_reach = reach_calculator.calculate_gross_reach()
    array_unique_reach = reach_calculator.calculate_unique_reach()

    return (
        array_gross_reach,
        array_unique_reach,
    )


def run_visualize_reach_curve(
    hill_nls: np.ndarray,
    log_nls: np.ndarray,
    array_gross_reach: np.ndarray,
    array_unique_reach: np.ndarray,
) -> None:
    x = np.linspace(0, max(array_gross_reach), 100)
    hill_y = hill_function(x, hill_nls[0], hill_nls[1])
    log_y = log_function(x, log_nls[0])

    plt.figure(figsize=(5, 3))
    plt.plot(x, hill_y, label="hill function", linestyle="-", alpha=0.75)
    plt.plot(x, log_y, label="log function", linestyle="--", alpha=0.75)
    plt.scatter(array_gross_reach, array_unique_reach, s=20, alpha=0.75)

    plt.xlabel("Gross reach (%)")
    plt.ylabel("Unique reach (%)")

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig("reach_curve.png", dpi=500)


def main():
    # ===== リーチデータ生成 =====
    df_reach = run_generate_reach_data(
        num_of_panels=NUM_OF_PANELS,
        num_of_cm_logs=NUM_OF_CM_LOGS,
        not_reach_ratio=NOT_REACH_RATIO,
        a_for_beta_distribution=A_FOR_BETA_DISTRIBUTION,
        b_for_beta_distribution=B_FOR_BETA_DISTRIBUTION,
        seed=SEED,
    )

    # ===== リーチ計算 =====
    array_gross_reach, array_unique_reach = run_calculate_reach(df_reach)

    # ===== hill functionのフィッティング =====
    hill_nls, _ = curve_fit(
        hill_function,
        array_gross_reach,
        array_unique_reach,
        bounds=(
            (0, -np.inf),
            (100, np.inf),
        ),
    )
    logger.info(f"{hill_nls=}")

    # ===== log functionのフィッティング =====
    log_nls, _ = curve_fit(
        log_function,
        array_gross_reach,
        array_unique_reach,
    )
    logger.info(f"{log_nls=}")

    # ===== 可視化 =====
    run_visualize_reach_curve(
        hill_nls,
        log_nls,
        array_gross_reach,
        array_unique_reach,
    )


if __name__ == "__main__":
    main()
