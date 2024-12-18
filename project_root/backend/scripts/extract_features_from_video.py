# import os
# import cv2
# import mysql.connector as mydb
# import pandas as pd
# import numpy as np
# from gensim import corpora
# from gensim.models import LdaModel
# import configparser
# import spacy
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from category_encoders import TargetEncoder
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from joblib import load
# import pickle
# import joblib
# from gensim import corpora
# from gensim.models import LdaModel
# import spacy
# import pytesseract
# from pytesseract import Output
# import logging
# from datetime import datetime

# # ----------------------------------------
# # ログの設定
# # ----------------------------------------
# # ログのフォーマットとレベルを設定
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# # 必要なモジュールのインポート
# from buzzAI import numeric_columns, text_columns, date_columns
# from video_analysis import extract_features, preprocess_text, get_topic_distribution, extract_text_from_video, analyze_sentiment

# # ----------------------------------------
# # ステップ 1: 設定ファイルの読み込みとデータベース接続
# # ----------------------------------------
# logging.info("ステップ 1: 設定ファイルの読み込みとデータベース接続")

# logging.info("Loading configuration file...")
# config = configparser.ConfigParser()
# config.read('/Users/p10475/BuzzCity/config.ini')

# logging.info("Connecting to the database...")
# conn = mydb.connect(
#     host=config['database']['host'],
#     port=int(config['database']['port']),
#     user=config['database']['user'],
#     password=config['database']['password'],
#     database=config['database']['database']
# )
# cursor = conn.cursor()

# # ----------------------------------------
# # ステップ 2: 必要なモデルや辞書のロード
# # ----------------------------------------
# logging.info("ステップ 2: 必要なモデルや辞書のロード")

# logging.info("Loading NLP model...")
# nlp = spacy.load("ja_core_news_sm")

# logging.info("Loading dictionary and LDA model...")
# # 辞書とLDAモデルのロード
# with open('dictionary.pkl', 'rb') as f:
#     dictionary = pickle.load(f)
# lda_model = LdaModel.load('lda_model.gensim')

# logging.info("Loading trained prediction model...")
# # 学習済みの予測モデルのロード
# model = joblib.load('trained_model.pkl')

# # スケーラーのロード（必要な場合）
# # from sklearn.preprocessing import StandardScaler
# # scaler = joblib.load('scaler.pkl')

# # ----------------------------------------
# # ステップ 3: 新しい動画のパスの指定と特徴量の抽出
# # ----------------------------------------
# logging.info("ステップ 3: 新しい動画のパスの指定と特徴量の抽出")

# # 新しいテスト動画のフォルダパス
# new_videos_dir = '/Users/p10475/BuzzCity/tiktok_testvideo'

# # 動画ファイルのリストを取得
# video_files = [os.path.join(new_videos_dir, f) for f in os.listdir(new_videos_dir) if f.endswith('.mp4')]
# logging.info(f"Found {len(video_files)} video files in {new_videos_dir}")

# # 新しい動画のDataFrameを作成
# new_video_data = pd.DataFrame({'video_id': range(len(video_files)), 'video_path': video_files})

# # フレーム数を取得する関数
# def get_frame_count(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logging.error(f"Error opening video file: {video_path}")
#         return 0
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return frame_count

# logging.info("Calculating frame counts for new videos...")
# new_video_data['frame_count'] = new_video_data['video_path'].apply(get_frame_count)

# # 特徴量の抽出
# logging.info("Extracting features from new videos...")
# new_video_features = new_video_data.apply(
#     lambda row: extract_features(row['video_path'], row['frame_count'], dictionary, lda_model), axis=1)

# new_video_features_df = pd.DataFrame(new_video_features.tolist())

# # ----------------------------------------
# # ステップ 4: 学習済みモデルのロードと予測の実行
# # ----------------------------------------
# logging.info("ステップ 4: 学習済みモデルのロードと予測の実行")

# # 特徴量のカラム名（学習時と同じ順序で指定）
# feature_columns = [
#     'num_texts', 'avg_size', 'avg_color_r', 'avg_color_g', 'avg_color_b',
#     'frame_count', 'sentiment',
#     'topic_0', 'topic_1', 'topic_2', 'topic_3',
#     'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9'
# ]

# # 新しい動画の特徴量データ
# X_new = new_video_features_df[feature_columns]

# # スケーリングの適用（必要な場合）
# # X_new = scaler.transform(X_new)

# # 視聴数の予測
# logging.info("Predicting view counts for new videos...")
# predictions = model.predict(X_new)

# # 予測結果をDataFrameに追加
# new_video_features_df['predicted_view_count'] = predictions
# new_video_features_df['video_path'] = new_video_data['video_path']
# new_video_features_df['video_id'] = new_video_data['video_id']

# # ----------------------------------------
# # ステップ 5: 予測結果の保存と表示
# # ----------------------------------------
# logging.info("ステップ 5: 予測結果の保存と表示")

# # 'predicted_views' テーブルの存在確認と作成
# cursor.execute("SHOW TABLES LIKE 'predicted_views'")
# table_exists = cursor.fetchone()

# if not table_exists:
#     logging.info("Creating 'predicted_views' table in the database...")
#     cursor.execute('''
#         CREATE TABLE predicted_views (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             video_id BIGINT,
#             predicted_view_count FLOAT,
#             video_path VARCHAR(255)
#         )
#     ''')
#     conn.commit()

# # 予測結果をデータベースに挿入
# logging.info("Inserting prediction results into the database...")
# # 動画の特徴量を抽出し、データベースに保存 (上書き)
# for index, row in new_video_data.iterrows():
#     video_id = row['video_id']
#     video_path = row['video_path']
#     frame_count = row['frame_count']

#     # 特徴量抽出
#     features = extract_features(video_path, frame_count, dictionary, lda_model)

#     # 新しいカラムをテーブルに追加（必要な場合）
#     new_columns = [
#         'audio_sentiment',
#         'audio_feature_0',
#         'audio_feature_1',
#         'audio_feature_2',
#         'audio_feature_3',
#         'audio_feature_4',
#         'audio_feature_5',
#         'audio_feature_6',
#         'audio_feature_7',
#         'audio_feature_8',
#         'audio_feature_9',
#         'audio_feature_10',
#         'audio_feature_11',
#         'audio_feature_12'
#     ]

#     existing_columns_query = "SHOW COLUMNS FROM video_features"
#     cursor.execute(existing_columns_query)
#     existing_columns = [column[0] for column in cursor.fetchall()]

#     for col in new_columns:
#         if col not in existing_columns:
#             alter_table_query = f"ALTER TABLE video_features ADD COLUMN {col} FLOAT"
#             cursor.execute(alter_table_query)
#             print(f"Added column '{col}' to 'video_features' table.")
            

#     # SQLクエリを実行し、video_featuresテーブルに情報を上書き保存
#     insert_query = """
#     REPLACE INTO video_features (
#         video_id,
#         num_texts,
#         avg_size,
#         avg_color_r,
#         avg_color_g,
#         avg_color_b,
#         keywords,
#         sentiment,
#         audio_sentiment,
#         topic_0,
#         topic_1,
#         topic_2,
#         topic_3,
#         topic_4,
#         topic_5,
#         topic_6,
#         topic_7,
#         topic_8,
#         topic_9,
#         audio_feature_0,
#         audio_feature_1,
#         audio_feature_2,
#         audio_feature_3,
#         audio_feature_4,
#         audio_feature_5,
#         audio_feature_6,
#         audio_feature_7,
#         audio_feature_8,
#         audio_feature_9,
#         audio_feature_10,
#         audio_feature_11,
#         audio_feature_12
#     )
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """

#     params = (
#         video_id,
#         features['num_texts'],
#         features['avg_size'],
#         features['avg_color_r'],
#         features['avg_color_g'],
#         features['avg_color_b'],
#         features['keywords'],
#         features['sentiment'],
#         features['audio_sentiment'],
#         features['topic_0'],
#         features['topic_1'],
#         features['topic_2'],
#         features['topic_3'],
#         features['topic_4'],
#         features['topic_5'],
#         features['topic_6'],
#         features['topic_7'],
#         features['topic_8'],
#         features['topic_9'],
#         features['audio_feature_0'],
#         features['audio_feature_1'],
#         features['audio_feature_2'],
#         features['audio_feature_3'],
#         features['audio_feature_4'],
#         features['audio_feature_5'],
#         features['audio_feature_6'],
#         features['audio_feature_7'],
#         features['audio_feature_8'],
#         features['audio_feature_9'],
#         features['audio_feature_10'],
#         features['audio_feature_11'],
#         features['audio_feature_12']
#     )

#     cursor.execute(insert_query, params)
#     conn.commit()

#     print(f"Processed video: {video_path}, features inserted into database.")

# # データベース接続を閉じる
# conn.close()

# # 結果をCSVファイルに保存
# output_dir = '/Users/p10475/BuzzCity/result'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# output_file = os.path.join(output_dir, 'new_video_predictions_test.csv')

# logging.info(f"Saving prediction results to '{output_file}'...")
# new_video_features_df.to_csv(output_file, index=False)

# # 結果の表示
# logging.info("Prediction results:")
# print(new_video_features_df[['video_path', 'predicted_view_count']])

# logging.info("All processes completed successfully.")

import os
import sys
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
import configparser
import spacy
from joblib import load
import pickle
import joblib
import pytesseract
from pytesseract import Output
import logging
from datetime import datetime
from gensim import corpora
from gensim.models import LdaModel

# スクリプトのディレクトリの親ディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root_dir)

# ----------------------------------------
# ログの設定
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 必要なモジュールのインポート
from video_analysis import preprocess_text, get_topic_distribution, extract_text_from_video, analyze_sentiment

# ----------------------------------------
# ステップ 1: 設定ファイルの読み込みとデータベース接続
# ----------------------------------------
logging.info("ステップ 1: 設定ファイルの読み込みとデータベース接続")

logging.info("Loading configuration file...")
config = configparser.ConfigParser()
config.read('/Users/p10475/BuzzCity/project_root/backend/config.ini')

logging.info("Connecting to the database...")
conn = mydb.connect(
    host=config['database']['host'],
    port=int(config['database']['port']),
    user=config['database']['user'],
    password=config['database']['password'],
    database=config['database']['database'],
    charset='utf8mb4'
)
cursor = conn.cursor()

# ----------------------------------------
# ステップ 2: 必要なモデルや辞書のロード
# ----------------------------------------
logging.info("ステップ 2: 必要なモデルや辞書のロード")

logging.info("Loading NLP model...")
nlp = spacy.load("ja_core_news_sm")

logging.info("Loading dictionary and LDA model...")
# 辞書とLDAモデルのロード
dictionary = corpora.Dictionary.load('dictionary.gensim')
lda_model = LdaModel.load('lda_model.gensim')

# 学習済みの予測モデルのロード
logging.info("Loading trained prediction models...")
rf_model = joblib.load('rf_model.joblib')
gb_model = joblib.load('gb_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')

# ----------------------------------------
# ステップ 3: 特徴量抽出関数の定義
# ----------------------------------------
def extract_features(video_path, frame_count, dictionary, lda_model):
    logging.info(f"Extracting text from video: {video_path}")
    text_data = extract_text_from_video(video_path)
    num_texts = len(text_data)
    avg_size = float(np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0)
    avg_color = [float(c) for c in (np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0])]

    combined_text = ' '.join([t['text'] for t in text_data])
    keywords = preprocess_text(combined_text)
    sentiment = float(analyze_sentiment(combined_text) if combined_text else 0)
    topic_distribution = get_topic_distribution(combined_text, dictionary, lda_model) if combined_text else [0]*10

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    features = {
        'video_id': video_id,
        'num_texts': num_texts,
        'avg_size': avg_size,
        'avg_color_r': avg_color[0],
        'avg_color_g': avg_color[1],
        'avg_color_b': avg_color[2],
        'frame_count': frame_count,
        'keywords': ' '.join(keywords),
        'sentiment': sentiment,
        **{f'topic_{i}': float(topic_distribution[i]) if i < len(topic_distribution) else 0 for i in range(10)}
    }

    return features

# ----------------------------------------
# ステップ 4: 新しい動画のパスの指定と特徴量の抽出
# ----------------------------------------
logging.info("ステップ 3: 新しい動画のパスの指定と特徴量の抽出")

new_videos_dir = '/Users/p10475/BuzzCity/project_root/backend/data/raw_videos/tiktok_testvideo'

video_files = [os.path.join(new_videos_dir, f) for f in os.listdir(new_videos_dir) if f.endswith('.mp4')]
logging.info(f"Found {len(video_files)} video files in {new_videos_dir}")

new_video_data = pd.DataFrame({'video_id': range(len(video_files)), 'video_path': video_files})

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

logging.info("Calculating frame counts for new videos...")
new_video_data['frame_count'] = new_video_data['video_path'].apply(get_frame_count)

# 特徴量の抽出
logging.info("Extracting features from new videos...")
new_video_features = new_video_data.apply(
    lambda row: extract_features(row['video_path'], row['frame_count'], dictionary, lda_model), axis=1)

new_video_features_df = pd.DataFrame(new_video_features.tolist())

# ----------------------------------------
# ステップ 5: 予測の実行
# ----------------------------------------
logging.info("ステップ 4: 学習済みモデルのロードと予測の実行")

feature_columns = [
    'num_texts', 'avg_size', 'avg_color_r', 'avg_color_g', 'avg_color_b',
    'frame_count', 'sentiment',
    'topic_0', 'topic_1', 'topic_2', 'topic_3',
    'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9'
]

X_new = new_video_features_df[feature_columns].astype(float)

# 各モデルで視聴数の予測
logging.info("Predicting view counts for new videos...")
new_video_features_df['rf_predicted_view_count'] = rf_model.predict(X_new)
new_video_features_df['gb_predicted_view_count'] = gb_model.predict(X_new)
new_video_features_df['xgb_predicted_view_count'] = xgb_model.predict(X_new)

new_video_features_df['video_path'] = new_video_data['video_path']
new_video_features_df['video_id'] = new_video_data['video_id']

# ----------------------------------------
# ステップ 6: 予測結果の保存と表示
# ----------------------------------------
logging.info("ステップ 5: 予測結果の保存と表示")

# 'predicted_views' テーブルの存在確認と作成
cursor.execute("SHOW TABLES LIKE 'predicted_views'")
table_exists = cursor.fetchone()

if not table_exists:
    logging.info("Creating 'predicted_views' table in the database...")
    cursor.execute('''
        CREATE TABLE predicted_views (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_id VARCHAR(255),
            rf_predicted_view_count FLOAT,
            gb_predicted_view_count FLOAT,
            xgb_predicted_view_count FLOAT,
            video_path VARCHAR(255)
        ) CHARACTER SET utf8mb4
    ''')
    conn.commit()

# 予測結果をデータベースに挿入
logging.info("Inserting prediction results into the database...")
for index, row in new_video_features_df.iterrows():
    insert_query = """
    REPLACE INTO predicted_views (
        video_id,
        rf_predicted_view_count,
        gb_predicted_view_count,
        xgb_predicted_view_count,
        video_path
    ) VALUES (%s, %s, %s, %s, %s)
    """
    params = (
        row['video_id'],
        row['rf_predicted_view_count'],
        row['gb_predicted_view_count'],
        row['xgb_predicted_view_count'],
        row['video_path']
    )
    cursor.execute(insert_query, params)
    conn.commit()

# 'video_features' テーブルの存在確認と作成
cursor.execute("SHOW TABLES LIKE 'video_features'")
table_exists = cursor.fetchone()

if not table_exists:
    logging.info("Creating 'video_features' table in the database...")
    create_table_query = '''
        CREATE TABLE video_features (
            video_id VARCHAR(255) PRIMARY KEY,
            num_texts INT,
            avg_size FLOAT,
            avg_color_r FLOAT,
            avg_color_g FLOAT,
            avg_color_b FLOAT,
            keywords TEXT,
            sentiment FLOAT,
            topic_0 FLOAT,
            topic_1 FLOAT,
            topic_2 FLOAT,
            topic_3 FLOAT,
            topic_4 FLOAT,
            topic_5 FLOAT,
            topic_6 FLOAT,
            topic_7 FLOAT,
            topic_8 FLOAT,
            topic_9 FLOAT
        ) CHARACTER SET utf8mb4
    '''
    cursor.execute(create_table_query)
    conn.commit()

# 特徴量をデータベースに挿入
logging.info("Inserting video features into the database...")
for index, row in new_video_features_df.iterrows():
    try:
        insert_query = """
        REPLACE INTO video_features (
            video_id,
            num_texts,
            avg_size,
            avg_color_r,
            avg_color_g,
            avg_color_b,
            keywords,
            sentiment,
            topic_0,
            topic_1,
            topic_2,
            topic_3,
            topic_4,
            topic_5,
            topic_6,
            topic_7,
            topic_8,
            topic_9
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            row['video_id'],
            row['num_texts'],
            row['avg_size'],
            row['avg_color_r'],
            row['avg_color_g'],
            row['avg_color_b'],
            row['keywords'],
            row['sentiment'],
            row['topic_0'],
            row['topic_1'],
            row['topic_2'],
            row['topic_3'],
            row['topic_4'],
            row['topic_5'],
            row['topic_6'],
            row['topic_7'],
            row['topic_8'],
            row['topic_9']
        )
        cursor.execute(insert_query, params)
        conn.commit()
        logging.info(f"Inserted features for video_id {row['video_id']} into database.")
    except Exception as e:
        logging.error(f"Error inserting features for video_id {row['video_id']}: {e}")

# データベース接続を閉じる
conn.close()

# 結果をCSVファイルに保存
output_dir = '/Users/p10475/BuzzCity/result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'new_video_predictions_test.csv')

logging.info(f"Saving prediction results to '{output_file}'...")
new_video_features_df.to_csv(output_file, index=False)

# 結果の表示
logging.info("Prediction results:")
print(new_video_features_df[['video_path', 'rf_predicted_view_count', 'gb_predicted_view_count', 'xgb_predicted_view_count']])

logging.info("All processes completed successfully.")

