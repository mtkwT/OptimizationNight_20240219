import numpy as np


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
