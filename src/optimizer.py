import pandas as pd
import numpy as np

from scipy.optimize import minimize

from src.curve_function import hill_function


class TotalReachOptimizer:
    def __init__(
        self,
        nls_params_dict: dict[str, np.ndarray],
        dict_gross_reach_cost: dict[str, int | float],
        total_budget: int | float,
    ):
        """
        Args:
            nls_params_dict (dict[str, np.ndarray]): 放送局ごとのhill関数のパラメータ
            {
                "A": array([\beta, \alpha]),
                "B": array([\beta, \alpha]),
                "C": array([\beta, \alpha]),
                "D": array([\beta, \alpha]),
                "E": array([\beta, \alpha]),
            }
            dict_gross_reach_cost (dict[str, int | float]): 放送局ごとのGross Reachの単価
            total_budget (int | float): 全体の広告予算
        """
        self.nls_params_dict = nls_params_dict
        self.broadcaster_name_list = list(nls_params_dict.keys())

        self.dict_gross_reach_cost = dict_gross_reach_cost
        self.total_budget = total_budget

    def run_optimize(self):
        bounds = [
            (0, self.total_budget / gross_reach_cost)
            for gross_reach_cost in self.dict_gross_reach_cost.values()
        ]

        optimize_results = minimize(
            fun=self._objective_function,
            x0=[0] * len(self.broadcaster_name_list),
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": self._budget_constraint},
            ],
            method="SLSQP",
            tol=1e-8,
        )

        return optimize_results

    def _budget_constraint(
        self, array_gross_reach: np.ndarray | list[float]
    ) -> int | float:
        """予算制約

        Args:
            array_gross_reach (np.ndarray | list[float]): グロスリーチの配列またはリスト

        Returns:
            int | float: 予算制約の評価値
        """
        dict_gross_reach = self._build_dict_gross_reach(array_gross_reach)

        return (
            self.dict_gross_reach_cost["A"] * dict_gross_reach["A"]
            + self.dict_gross_reach_cost["B"] * dict_gross_reach["B"]
            + self.dict_gross_reach_cost["C"] * dict_gross_reach["C"]
            + self.dict_gross_reach_cost["D"] * dict_gross_reach["D"]
            + self.dict_gross_reach_cost["E"] * dict_gross_reach["E"]
            - self.total_budget
        )

    def _objective_function(
        self,
        array_gross_reach: np.ndarray | list[float],
    ):
        """Total Reachの関数
        \mathcal{B}: 放送局の集合
        1 - \prod_{b \in \mathcal{B}} (1 - R_{b}(g_{b}))

        Args:
            array_gross_reach (np.ndarray | list[float]): 放送局ごとのGross Reach
            A, B, C, D, Eの順番に格納されている
        """
        # ユニークリーチをhill関数で計算
        dict_gross_reach = self._build_dict_gross_reach(array_gross_reach)

        array_unique_reach = [
            hill_function(
                dict_gross_reach[broadcaster_name],
                self.nls_params_dict[broadcaster_name][0],
                self.nls_params_dict[broadcaster_name][1],
            )
            for broadcaster_name in self.broadcaster_name_list
        ]

        # Total Reachを計算
        total_reach = 1 - (
            (1 - array_unique_reach[0] / 100)
            * (1 - array_unique_reach[1] / 100)
            * (1 - array_unique_reach[2] / 100)
            * (1 - array_unique_reach[3] / 100)
            * (1 - array_unique_reach[4] / 100)
        )

        return -100 * total_reach

    def _build_dict_gross_reach(
        self, array_gross_reach: np.ndarray | list[float]
    ) -> dict[str, int | float]:
        """numpy配列またはリスト形式グロスリーチの順番を放送局名に対応させる

        Args:
            array_gross_reach (np.ndarray | list[float]): グロスリーチの配列またはリスト

        Returns:
            dict[str, int | float]: 放送局名をキーとしたグロスリーチの辞書
        """
        return dict(
            zip(
                self.broadcaster_name_list,
                array_gross_reach,
                strict=True,
            )
        )

    def post_process(
        self,
        optimize_results,
    ) -> pd.DataFrame:
        """最適化結果を整形してDataFrameに格納

        Returns:
            pd.DataFrame: 最適化結果のDataFrame
        """

        dict_gross_reach = self._build_dict_gross_reach(optimize_results.x)
        dict_gross_reach["Total"] = sum(dict_gross_reach.values())
        dict_unique_reach = {
            broadcaster_name: hill_function(
                dict_gross_reach[broadcaster_name],
                self.nls_params_dict[broadcaster_name][0],
                self.nls_params_dict[broadcaster_name][1],
            )
            for broadcaster_name in self.broadcaster_name_list
        }
        dict_unique_reach["Total"] = -optimize_results.fun

        dict_broadcast_cost = {
            broadcaster_name: dict_gross_reach[broadcaster_name]
            * self.dict_gross_reach_cost[broadcaster_name]
            for broadcaster_name in self.broadcaster_name_list
        }
        dict_broadcast_cost["Total"] = sum(dict_broadcast_cost.values())

        df_optimize_results = pd.DataFrame(
            {
                "Broadcaster": list(dict_gross_reach.keys()),
                "Gross Reach(%)": list(dict_gross_reach.values()),
                "Unique Reach(%)": list(dict_unique_reach.values()),
                "Broadcast Cost(万円)": list(dict_broadcast_cost.values()),
            }
        )

        return df_optimize_results
