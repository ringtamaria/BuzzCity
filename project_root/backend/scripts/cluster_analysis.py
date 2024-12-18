import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def perform_clustering(merged_data):
    # データの前処理
    # 必要に応じて欠損値処理や特徴量エンジニアリングなどを行う
    # ...

    # 特徴量を数値データに変換（例：One-Hot Encoding）
    # ...

    # 教師なし学習 (例: K-meansクラスタリング)
    X = merged_data.drop(['video_id', 'video_path', 'id'], axis=1)  # video_idとvideo_pathは学習には不要
    kmeans = KMeans(n_clusters=5, random_state=0)  # 5つのクラスタに分類
    cluster_labels = kmeans.fit_predict(X)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data['実績再生回数'], merged_data['frame_count'], c=cluster_labels, cmap='viridis')
    plt.xlabel('実績再生回数')
    plt.ylabel('フレーム数')
    plt.title('TikTok動画クラスタリング結果')

    # 各動画の情報をグラフ上に表示
    for i, row in merged_data.iterrows():
        plt.annotate(f"{row['video_id']}", (row['実績再生回数'], row['frame_count']), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()
