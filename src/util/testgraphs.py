import networkx as nx
import numpy as np
import random
import pandas as pd
from typing import Tuple, Union

XLIM_DEFAULT = 1000
YLIM_DEFAULT = 1000

def get_graph(name: Union[int, str], xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT):
    """
    サンプルグラフの取得関数。
    Args:
        name: グラフタイプ（intまたはstr）
        xlim: 描画範囲x
        ylim: 描画範囲y
    Returns:
        (G, pos) or (G, pos, param) 形式
    """
    mapping = {
        1: graph1,
        2: graph2,
        3: graph3,
        4: graph4,
        5: graph5,
        "jpair": japan,
        "usair": usairline,
        "air": airline,
    }
    if name not in mapping:
        raise ValueError(f"Unknown graph name: {name}")
    func = mapping[name]
    XLIM_DEFAULT = xlim
    YLIM_DEFAULT = ylim
    return func()

def graph1(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT, yrang: float = 0.2, yplace: float = 0.28, xrang: float = 0.2, n_n: int = 100) -> Tuple[nx.Graph, np.ndarray]:
    """
    平行なサンプルグラフ
    Returns:
        G: networkx.Graph
        pos: np.ndarray [n_n, 2]
    """
    if n_n % 4 != 0:
        raise ValueError("n_n must be a multiple of 4.")
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((i, i + 1) for i in range(0, n_n, 2))

    random.seed(1)
    pos = np.empty((n_n, 2))
    for i in range(0, n_n, 4):
        pos[i] = [random.random() * xlim * xrang, (random.random() * yrang + yplace) * ylim]
        pos[i + 1] = [xlim - (random.random() * xlim * xrang), (random.random() * yrang + yplace) * ylim]
        pos[i + 2] = [random.random() * xlim * xrang, ylim - ((random.random() * yrang + yplace) * ylim)]
        pos[i + 3] = [xlim - (random.random() * xlim * xrang), ylim - ((random.random() * yrang + yplace) * ylim)]
    return G, pos, (xlim, ylim, 10, 10)

def graph2(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    三角形のサンプルグラフ
    """
    n_n = 101
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((0, i) for i in range(1, n_n))

    pos = np.empty((n_n, 2))
    pos[0] = [xlim / 2, 0]
    d = xlim / (n_n - 1)
    for i in range(1, n_n):
        pos[i] = [d * (i - 1), ylim]
    return G, pos, (xlim, ylim, 10, 10)

def graph3(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    クロス型のサンプルグラフ
    """
    n_n = 100
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((i, n_n - (i + 1)) for i in range(n_n // 2))

    d = xlim / n_n * 2
    pos = np.empty((n_n, 2))
    for i in range(n_n // 2):
        pos[i] = [d * i, 0]
        pos[n_n - (i + 1)] = [xlim - d * i, ylim]
    return G, pos, (xlim, ylim, 10, 10)

def graph4(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    三角形が二つならんだサンプルグラフ
    """
    n_n = 102
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((0, i) for i in range(1, n_n // 2))
    G.add_edges_from((n_n // 2, i) for i in range(n_n // 2 + 1, n_n))

    pos = np.empty((n_n, 2))
    x_temp = 0
    d = xlim // (n_n - 2)
    pos[0] = [xlim // 4, 0]
    pos[n_n // 2] = [xlim // 4 * 3, 0]
    for i in range(n_n):
        if i not in (0, n_n // 2):
            pos[i] = [x_temp, ylim]
            x_temp += d
    return G, pos, (xlim, ylim, 10, 10)

def graph5(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    4つの辺からなる閉路状のサンプルグラフ
    """
    n_n = 100
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    for i in range(n_n):
        if i not in (0, n_n // 2):
            G.add_edge(i, n_n - i)

    pos = np.empty((n_n, 2))
    dx = xlim // (n_n // 4)
    dy = ylim // (n_n // 4)
    for i in range(0, n_n // 4):
        pos[i] = [0, i * dy]
    for i in range(n_n // 4, n_n // 2):
        pos[i] = [(i - n_n / 4) * dx, ylim]
    for i in range(n_n // 2, n_n // 4 * 3):
        pos[i] = [xlim, ylim - (i - n_n / 2) * dy]
    for i in range(n_n // 4 * 3, n_n):
        pos[i] = [xlim - (i - n_n / 4 * 3) * dx, 0]
    return G, pos, (xlim, ylim, 10, 10)

def airline(node_file: str = "../data_set/Node_AirPorts.csv",
            edge_file: str = "../data_set/Edge_AirPorts.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    世界の航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 1590, 530, 9, 3
    G, pos = _load_graph_from_csv(node_file, edge_file, swap_xy=True)
    return G, pos, (xlim, ylim, x_show, y_show)

def usairline(node_file: str = "../data_set/Node_AmericaAirLines.csv", edge_file: str = "../data_set/Edge_AmericaAirLines.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    アメリカの航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 700, 350, 10, 5
    G, pos = _load_graph_from_csv(node_file, edge_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def japan(node_file: str = "../data_set/Node_Japan.csv", edge_file: str = "../data_set/Edge_Japan.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    日本の航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 800, 800, 10, 10
    G, pos = _load_graph_from_csv(node_file, edge_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def _load_graph_from_csv(node_file: str, edge_file: str, swap_xy: bool = False) -> Tuple[nx.Graph, np.ndarray]:
    """
    CSVファイルからグラフと座標を読み込む
    """
    G = nx.Graph()
    d1 = pd.read_csv(node_file)
    d2 = pd.read_csv(edge_file)
    G.add_nodes_from(range(len(d1)))
    G.add_edges_from((int(d2.loc[i, 'source']) - 1, int(d2.loc[i, 'target']) - 1) for i in range(len(d2)))
    if swap_xy:
        pos = np.array([[d1.loc[i, 'y'], d1.loc[i, 'x']] for i in range(len(d1))])
    else:
        pos = np.array([[d1.loc[i, 'x'], d1.loc[i, 'y']] for i in range(len(d1))])
    return G, pos
