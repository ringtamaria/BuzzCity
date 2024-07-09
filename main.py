import configparser
import mysql.connector as mydb
import pandas as pd
import matplotlib.pyplot as plt
from video_analysis import extract_features_from_videos
from cluster_analysis import perform_clustering

# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read('config.ini')

# データベース接続
conn = mydb.connect(
    host=config['database']['host'],
    port=config['database']['port'],
    user=config['database']['user'],
    password=config['database']['password'],
    database=config['database']['database']
)
cursor = conn.cursor()

# データベースから動画情報を読み込み
cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

# データベースからCSVデータを読み込み
cursor.execute("SELECT * FROM numeric_data")
csv_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# 動画から特徴量を抽出
video_features_df = extract_features_from_videos(video_data)

# データ結合
merged_data = pd.merge(video_data[['video_id', 'frame_count']], csv_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.concat([merged_data, video_features_df], axis=1)

# クラスタリングの実行と可視化
perform_clustering(merged_data)

# データベース接続を閉じる
conn.close()

