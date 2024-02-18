import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from scipy.optimize import curve_fit
from loguru import logger

from visualize_reach_curve import (
    run_generate_reach_data,
    run_calculate_reach,
)
from src.curve_function import hill_function
from src.optimizer import TotalReachOptimizer

# シミュレーションのパラメータ
NUM_OF_PANELS = 1000
NUM_OF_CM_LOGS = 26
# 放送局ごとに、リーチカーブの上限が変わるようにしたいので、
# 全くリーチしない割合を放送局ごとに設定する。値は適当である。
NOT_REACH_RATIO_DICT = {
    "A": 0.15,  # 上限は85%くらいになる
    "B": 0.2,  # 上限は80%くらいになる
    "C": 0.25,  # 上限は75%くらいになる
    "D": 0.38,  # 上限は62%くらいになる
    "E": 0.45,  # 上限は55%くらいになる
}

A_FOR_BETA_DISTRIBUTION = 1
B_FOR_BETA_DISTRIBUTION = 5
SEED = 42


def run_visualize_reach_curve_all_broadcaster(
    nls_params_dict: dict[str, np.ndarray],
    max_gross_reach: float,
    save_file_path: str,
) -> None:
    x = np.linspace(0, max_gross_reach, 100)

    plt.figure(figsize=(5, 2.5))
    for broadcaster_name, hill_nls in nls_params_dict.items():
        hill_y = hill_function(x, hill_nls[0], hill_nls[1])
        plt.plot(x, hill_y, label=f"{broadcaster_name}局", linestyle="-", alpha=0.75)
    plt.xlabel("Gross reach (%)")
    plt.ylabel("Unique reach (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_file_path, dpi=350)


def main():
    # 放送局ごとにリーチデータを生成し、リーチカーブのパラメータを推定する
    nls_params_dict = {}

    for broadcaster_name, not_reach_ratio in NOT_REACH_RATIO_DICT.items():
        df_reach = run_generate_reach_data(
            num_of_panels=NUM_OF_PANELS,
            num_of_cm_logs=NUM_OF_CM_LOGS,
            not_reach_ratio=not_reach_ratio,
            a_for_beta_distribution=A_FOR_BETA_DISTRIBUTION,
            b_for_beta_distribution=B_FOR_BETA_DISTRIBUTION,
            seed=SEED,
        )

        array_gross_reach, array_unique_reach = run_calculate_reach(df_reach)

        hill_nls, _ = curve_fit(
            f=hill_function,
            xdata=array_gross_reach,
            ydata=array_unique_reach,
            bounds=(
                (0, -np.inf),
                (100, np.inf),
            ),
        )

        nls_params_dict[broadcaster_name] = hill_nls

    logger.info(f"{nls_params_dict=}")
    run_visualize_reach_curve_all_broadcaster(
        nls_params_dict=nls_params_dict,
        max_gross_reach=400,
        save_file_path="reach_curve_all_broadcaster.png",
    )

    # 放送局ごとのGross Reachの単価と全体の広告予算を設定し、最適化を実行する
    # 単位は万円とする
    dict_gross_reach_cost = {
        "A": 12.5,
        "B": 12,
        "C": 11,
        "D": 8,
        "E": 7.5,
    }
    total_budget = 2000

    optimizer = TotalReachOptimizer(
        nls_params_dict=nls_params_dict,
        dict_gross_reach_cost=dict_gross_reach_cost,
        total_budget=total_budget,
    )
    optimize_results = optimizer.run_optimize()
    logger.info(f"{optimize_results=}")

    df_optimize_results = optimizer.post_process(optimize_results)
    df_optimize_results.to_excel("optimize_results.xlsx", index=False)


if __name__ == "__main__":
    main()
