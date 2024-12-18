import os
import cv2
import mysql.connector as mydb
import configparser
from . import buzzAI

# プロジェクトルートディレクトリを取得
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# config.ini のパスを設定
config_path = os.path.join(project_root, 'config.ini')

# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')
print(f"Config file path: {config_path}")
print(f"Sections in config file: {config.sections()}")

# 動画ファイルのディレクトリパス
video_dir = '/Users/p10475/BuzzCity/project_root/backend/data/raw_videos/tiktok_video'

# データベース接続
try:
    conn = mydb.connect(
        host=config['database']['host'],
        port=int(config['database']['port']),
        user=config['database']['user'],
        password=config['database']['password'],
        database=config['database']['database']
    )
    cursor = conn.cursor()
except mydb.Error as err:
    print(f"Database connection error: {err}")
    exit(1)

# テーブル作成 (存在しない場合)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INT PRIMARY KEY AUTO_INCREMENT,
        video_id INT,
        video_path TEXT,
        frame_count INT
    )
''')

# 動画ファイルのリストを取得
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# IDの最小値を取得 (buzzAI.pyのdataから)
min_video_id = int(buzzAI.data['id'].min())

# IDのリストを取得
video_ids_in_data = set(buzzAI.data['id'].astype(int).tolist())

for video_file in video_files:
    try:
        # ファイル名からIDを抽出
        video_id = int(video_file.split('.')[0])

        # buzzAI.pyのdataからIDが存在するか確認
        if video_id in video_ids_in_data:
            # 動画ファイルのパス
            video_path = os.path.join(video_dir, video_file)

            # 動画を読み込む
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue

            # フレーム数をカウント
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # データベースのIDをvideo_idと一致させる
            db_id = int(video_id - min_video_id + 1)  # 明示的に int に変換

            # SQLクエリを実行し、データベースに情報を挿入または更新
            insert_query = """
            INSERT INTO videos (id, video_id, video_path, frame_count) 
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE video_path = VALUES(video_path), frame_count = VALUES(frame_count)
            """
            cursor.execute(insert_query, (
                db_id,
                video_id,
                video_path,
                frame_count
            ))
            conn.commit()

            print(f"Processed video: {video_file}, frame count: {frame_count}, inserted into database with ID: {db_id}")
        else:
            print(f"Video ID {video_id} not found in CSV data.")

    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        import traceback
        traceback.print_exc()

# データベース接続を閉じる
if conn.is_connected():
    cursor.close()
    conn.close()
    print("Connection closed")
