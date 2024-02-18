## 数理最適化: Optimization Night #8 での発表資料のシミュレーションコード
- https://optimization.connpass.com/event/308608/

## 最適化の実行
以下でOK
```bash
$ python optimize_total_reach.py
```

## ソースコード概要
- visualize_reach_curve.py
  - 単一のリーチデータ生成とリーチカーブ推定を行い、可視化する

- optimize_total_reach.py
  - 特定の設定で5局のトータルでのユニークリーチ最適化を実行する

- src/curve_function.py
  - hill関数と対数関数を定義

- src/simulator.py
  - 確率的シミュレーションでリーチデータの生成と、グロスリーチ・ユニークリーチ計算を行うクラス

- src/optimizer.py
  - 放送局ごとに推定されたhillのパラメータと、局ごとのパーコスト・トータルの予算を入れると、トータルのユニークリーチを最大化するクラス
