"""Utility class implementing GPU based metrics for edge bundling."""

from __future__ import annotations

import numpy as np
import cupy as cp

bresenham_kernel = cp.RawKernel(r'''
extern "C" __global__
void bresenham(const float* edges, int* grid, int width, int height, int edge_count, int nodes_per_edge, int cell_width, int cell_height) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;  // グローバルスレッドID
    if (edge_idx >= edge_count) return;

    // 対応するエッジの開始インデックス
    const float* edge = &edges[edge_idx * nodes_per_edge * 2];

    for (int i = 0; i < nodes_per_edge - 1; ++i) {
        float x0 = edge[i * 2];
        float y0 = edge[i * 2 + 1];
        float x1 = edge[(i + 1) * 2];
        float y1 = edge[(i + 1) * 2 + 1];

        int ix0 = int(round(x0));
        int iy0 = int(round(y0));
        int ix1 = int(round(x1));
        int iy1 = int(round(y1));

        int dx = abs(ix1 - ix0);
        int dy = abs(iy1 - iy0);
        int sx = (ix0 < ix1) ? 1 : -1;
        int sy = (iy0 < iy1) ? 1 : -1;
        int err = dx - dy;

        // 最初の点をスキップ
        bool skip_first = (i > 0);

        while (true) {
            if (!skip_first && ix0 >= 0 && ix0 < width && iy0 >= 0 && iy0 < height) {
                // 大グリッドのインデックスを計算
                int gx = ix0 / cell_width;
                int gy = iy0 / cell_height;

                // グリッド更新
                atomicAdd(&grid[gy * (width / cell_width) + gx], 1);
            }
            skip_first = false;  // 最初の点をスキップした後は通常通り描画

            if (ix0 == ix1 && iy0 == iy1) break;

            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; ix0 += sx; }
            if (e2 < dx) { err += dx; iy0 += sy; }
        }
    }
}
''', 'bresenham')

crossing_kernel = cp.RawKernel(r'''
extern "C" __device__
float orient(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

extern "C" __device__
bool segments_cross(float ax0, float ay0, float ax1, float ay1,
                    float bx0, float by0, float bx1, float by1) {
    float o1 = orient(ax0, ay0, ax1, ay1, bx0, by0);
    float o2 = orient(ax0, ay0, ax1, ay1, bx1, by1);
    float o3 = orient(bx0, by0, bx1, by1, ax0, ay0);
    float o4 = orient(bx0, by0, bx1, by1, ax1, ay1);
    return (o1 * o2 < 0.0f) && (o3 * o4 < 0.0f);
}

extern "C" __global__
void count_crossings(const float* edges, int n_edges, int nodes_per_edge, int* out_counts) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int cnt = 0;
    int npt = nodes_per_edge;
    int base1 = e * npt * 2;

    for (int other = e + 1; other < n_edges; ++other) {
        int base2 = other * npt * 2;
        for (int i = 0; i < npt - 1; ++i) {
            float ax0 = edges[base1 + i * 2];
            float ay0 = edges[base1 + i * 2 + 1];
            float ax1 = edges[base1 + (i + 1) * 2];
            float ay1 = edges[base1 + (i + 1) * 2 + 1];
            for (int j = 0; j < npt - 1; ++j) {
                float bx0 = edges[base2 + j * 2];
                float by0 = edges[base2 + j * 2 + 1];
                float bx1 = edges[base2 + (j + 1) * 2];
                float by1 = edges[base2 + (j + 1) * 2 + 1];
                if (segments_cross(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)) {
                    cnt += 1;
                }
            }
        }
    }

    out_counts[e] = cnt;
}
''', 'count_crossings')

class MyFunc:
    def __init__(self, G, pos, c, param = (1000, 1000, 10, 10), grid_size=1):
        self.n_n = len(G.nodes())
        self.n_e = len(G.edges())
        self.G = G
        self.pos = pos
        self.c = c
        self.xlim = param[0]
        self.ylim = param[1]
        self.x_show = param[2]
        self.y_show = param[3]
        self.size = (self.ylim, self.xlim)
        self.grid_size = grid_size
        self._threads_per_block = 512

    def total_Length(self, pos_np):
        """
        総延長を比較する
        """
        pos_tl = pos_np - np.roll(pos_np, 1, axis=1)
        pos_tl = np.sqrt(pos_tl[:, :, 0] ** 2 + pos_tl[:, :, 1] ** 2)
        return pos_tl[:, 1:].sum()


    def meld(self, pos_np, nomalization=True):
        """
        pos_np:
            改良版の座標配列。
        """
        pos_meld = pos_np - np.roll(pos_np, 1, axis=1)
        pos_meld = np.sqrt(pos_meld[:, :, 0] ** 2 + pos_meld[:, :, 1] ** 2)
        if nomalization:
            pos_meld = np.abs(1 - np.sum(pos_meld[:, 1:], axis=1) / pos_meld[:, 0])
        else:
            pos_meld = np.abs(np.sum(pos_meld[:, 1:], axis=1) - pos_meld[:, 0])
        return pos_meld.sum() / self.n_e
    
    def _edd(self, pos_np: np.ndarray) -> cp.ndarray:
        """Calculate edge density using a GPU-based Bresenham kernel."""
        grid_height = self.size[0] // self.grid_size
        grid_width = self.size[1] // self.grid_size
        grid = cp.zeros((grid_height, grid_width), dtype=cp.int32)
        edges_flat = cp.array(pos_np).reshape(-1).astype(cp.float32)
        _, num_points, _ = pos_np.shape
        blocks = (self.n_e + self._threads_per_block - 1) // self._threads_per_block
        bresenham_kernel(
            (blocks,),
            (self._threads_per_block,),
            (
                edges_flat,
                grid,
                self.size[1],
                self.size[0],
                self.n_e,
                num_points,
                self.grid_size,
                self.grid_size,
            ),
        )
        return grid
    def _moa(self, grid: cp.ndarray) -> float:
        """Calculate mean occupancy ratio of the density grid."""
        grid_temp = cp.where(grid != 0, 1, 0)
        return cp.sum(grid_temp) / grid.size

    def moa_edd(self, pos_np: np.ndarray) -> tuple[float, float]:
        """Return mean occupancy and squared edge density for positions."""
        grid = self._edd(pos_np)
        moa = self._moa(grid)
        ave = cp.sum(grid) / self.size[0] / self.size[1]
        grid = grid.astype(cp.float32)
        grid -= ave
        edd2 = cp.sum(cp.square(grid)).get() / self.size[0] / self.size[1]
        return moa, edd2
    
    def path_quality(self, pos_n: np.ndarray) -> float:
        """Return zig-zag metric of the given control points."""
        pos_np = pos_n.copy()
        pos_np -= np.roll(pos_np, 1, axis=1)
        # 偏角
        A = np.arctan2(pos_np[:, :, 1], pos_np[:, :, 0])
        A -= A[:, [0]]  # 始点方向で正規化
        # 曲がり差分
        delta_np = A - np.roll(A, 1, axis=1)
        # 角度補正を一発で
        delta_np = (delta_np + np.pi) % (2 * np.pi) - np.pi
        # ジグザグ判定
        gamma_np = np.sign(delta_np) != np.sign(np.roll(delta_np, 1, axis=1))
        gamma_np[:, :3] = 0  # 先頭3点
        # ジグザグ度
        values = gamma_np * np.abs(delta_np)
        values[np.abs(values) < 1e-14] = 0
        return np.sum(values)   


    def crossing_count_cuda(self, pos_np: np.ndarray) -> int:
        """Count edge intersections using a CUDA kernel.

        Parameters
        ----------
        pos_np : np.ndarray
            Edge coordinate array of shape ``(n_e, c+2, 2)``.
        """
        edges_flat = cp.array(pos_np, dtype=cp.float32).reshape(-1)
        counts = cp.zeros(self.n_e, dtype=cp.int32)
        _, num_points, _ = pos_np.shape
        blocks = (self.n_e + self._threads_per_block - 1) // self._threads_per_block
        crossing_kernel(
            (blocks,),
            (self._threads_per_block,),
            (
                edges_flat,
                self.n_e,
                num_points,
                counts,
            ),
        )
        return int(cp.sum(counts).get())

    def crossing_count_sweepline(self, pos_np: np.ndarray) -> int:
        """Count edge crossings using a simple sweep line algorithm.

        This implementation avoids checking all edge pairs by only
        comparing segments whose ``x`` ranges overlap. It yields the
        same result as :meth:`crossing_count` but typically runs faster
        than the brute-force approach.

        Parameters
        ----------
        pos_np : np.ndarray
            Edge coordinate array of shape ``(n_e, c+2, 2)``.
        """

        def _orient(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        def _intersects(s1, s2) -> bool:
            a1x, a1y, a2x, a2y, e1 = s1
            b1x, b1y, b2x, b2y, e2 = s2
            if e1 == e2:
                return False
            o1 = _orient(a1x, a1y, a2x, a2y, b1x, b1y)
            o2 = _orient(a1x, a1y, a2x, a2y, b2x, b2y)
            if o1 * o2 >= 0:
                return False
            o3 = _orient(b1x, b1y, b2x, b2y, a1x, a1y)
            o4 = _orient(b1x, b1y, b2x, b2y, a2x, a2y)
            return o3 * o4 < 0

        segs = []
        n_e, n_pt, _ = pos_np.shape
        for e in range(n_e):
            for i in range(n_pt - 1):
                x0, y0 = pos_np[e, i]
                x1, y1 = pos_np[e, i + 1]
                segs.append([x0, y0, x1, y1, e])

        events = []
        for idx, (x0, y0, x1, y1, _) in enumerate(segs):
            events.append((min(x0, x1), 0, idx))
            events.append((max(x0, x1), 1, idx))
        events.sort()

        active: list[int] = []
        count = 0
        for _, typ, idx in events:
            if typ == 0:  # segment enters sweep line
                for j in active:
                    if _intersects(segs[idx], segs[j]):
                        count += 1
                active.append(idx)
            else:  # segment leaves sweep line
                if idx in active:
                    active.remove(idx)
        return count
    
    # ---------------------------
    # Bezier helpers / wrappers
    # ---------------------------
    def sample_bezier(self, ctrl_pts: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Sample Bezier curves for each edge from control points.

        Parameters
        ----------
        ctrl_pts : np.ndarray
            Control points array shape (n_e, m, 2)
        n_samples : int
            Number of samples per edge

        Returns
        -------
        np.ndarray
            Sampled points shape (n_e, n_samples, 2)
        """
        n_e, m, _ = ctrl_pts.shape
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
        n = m - 1
        # Bernstein basis (m, n_samples)
        from math import comb
        B = np.vstack([comb(n, k) * (t**k) * ((1 - t)**(n - k)) for k in range(n + 1)])  # (m, n_samples)
        samples = np.empty((n_e, n_samples, 2), dtype=np.float64)
        for e in range(n_e):
            # ctrl_pts[e] -> (m,2), produce (2, n_samples) then transpose
            curve = ctrl_pts[e].T @ B  # (2, n_samples)
            samples[e] = curve.T
        return samples
