import os
import cv2
import mysql.connector as mydb
import buzzAI

# 動画ファイルのディレクトリパス
video_dir = '/Users/p10475/BuzzCity/tiktok_video'

# データベース接続
conn = mydb.connect(
    host="localhost",
    port='3306',
    user="rintamaria",
    password="buzzai",
    database="mydb"
)
cursor = conn.cursor()

# テーブル作成 (存在しない場合)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        video_id TEXT,
        video_path TEXT,
        frame_count INTEGER
    )
''')

# 動画ファイルのリストを取得
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# IDの最小値を取得 (buzzAI.pyのdataから)
min_video_id = buzzAI.data['id'].min()

for video_file in video_files:
    try:
        # ファイル名からIDを抽出
        video_id = int(video_file.split('.')[0])

        # buzzAI.pyのdataからIDが存在するか確認
        if video_id in buzzAI.data['id'].values:
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
            db_id = video_id - min_video_id + 1  # 最小のvideo_idからの差分を計算

            # SQLクエリを実行し、データベースに情報を挿入または更新
            insert_query = """
            INSERT INTO videos (id, video_id, video_path, frame_count) 
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE video_path = VALUES(video_path), frame_count = VALUES(frame_count)
            """
            cursor.execute(insert_query, (db_id, video_id, video_path, frame_count))
            conn.commit()

            print(f"Processed video: {video_file}, frame count: {frame_count}, inserted into database with ID: {db_id}")
        else:
            print(f"Video ID {video_id} not found in CSV data.")

    except Exception as e:
        print(f"Error processing {video_file}: {e}")

# データベース接続を閉じる
conn.close()
