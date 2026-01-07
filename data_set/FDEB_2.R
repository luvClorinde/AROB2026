# 必要なライブラリのロード
library(igraph)
library(edgebundle)
library(ggplot2)
library(jsonlite)

# CSVファイルの選択
cat("ノードデータのCSVファイルを選択してください。\n")
node_file <- file.choose()

cat("エッジデータのCSVファイルを選択してください。\n")
edge_file <- file.choose()

# ノードとエッジデータの読み込み
nodes <- read.csv(node_file)
edges <- read.csv(edge_file)
nodes$y <- -nodes$y  # y座標を反転

# グラフの構築
g <- graph_from_data_frame(d = edges, vertices = nodes, directed = FALSE)

# ノードの座標をそのまま利用
layout <- as.matrix(nodes[, c("x", "y")])  # x, y 列を行列に変換

# エッジバンドリング
bundled_edges <- edge_bundle_force(
  g, 
  xy = layout,
  K = 1,
  C = 6,
  P = 1,
  S = 0.04,
  P_rate = 2,
  I = 50,
  I_rate = 2 / 3,
  compatibility_threshold = 0.4,
  eps = 1e-8
)

# データフレームの作成
df_edges_bundled <- as.data.frame(bundled_edges)
df_nodes <- as.data.frame(layout)
colnames(df_nodes) <- c("x", "y")  # カラム名を "x", "y" に設定

# 可視化（アスペクト比を自動調整）
# 可視化（ノードを非表示、線の太さと透過率を変更）
ggplot() + 
  geom_path(data = df_edges_bundled, aes(x = x, y = y, group = group), 
            linewidth = 1.0,   # 線の太さ
            alpha = 0.8,
		color = "#000000") +  # 透過率（0が完全透明、1が不透明）
  coord_fixed(ratio = 1) +  # アスペクト比を固定（1:1）
  theme_void()

# ggsave("output.png", width = 10, height = 10, dpi = 300, bg = "#2F3438")
ggsave("output.png", width = 10, height = 10, dpi = 600, bg = "white")
#ggsave("output.png", width = 10, height = 10, dpi = 1500)