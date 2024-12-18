import os
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
import spacy
import configparser
import pytesseract
from pytesseract import Output
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# buzzAI.py から変数をインポート（必要に応じて）
from project_root.backend.scripts.buzzAI import numeric_columns, text_columns, date_columns

print("Loading configuration file...")
# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read('/Users/p10475/BuzzCity/project_root/backend/config.ini')

print("Connecting to the database...")
# データベース接続
conn = mydb.connect(
    host=config['database']['host'],
    port=config['database']['port'],
    user=config['database']['user'],
    password=config['database']['password'],
    database=config['database']['database']
)
cursor = conn.cursor()

print("Loading video data from the database...")
# データベースから動画情報を読み込み
cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

print("Loading numeric data from the database...")
# データベースから数値データを読み込み
cursor.execute("SELECT * FROM numeric_data")
numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

print("Loading text data from the database...")
# データベースからテキストデータを読み込み
cursor.execute("SELECT * FROM text_data")
text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

print("Loading date data from the database...")
# データベースから日付データを読み込み
cursor.execute("SELECT * FROM date_data")
date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

print("Loading NLP model...")
# NLPモデルの読み込み
nlp = spacy.load("ja_core_news_sm")

# テキストの前処理
def preprocess_text(text, chunk_size=45000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    tokens = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens.extend([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
    return tokens

# トピックモデリングの準備
def get_topic_distribution(text, dictionary, lda_model):
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)
    topic_distribution = lda_model.get_document_topics(bow)
    # トピックの確率をリストに変換
    topic_probs = [0] * lda_model.num_topics
    for topic_id, prob in topic_distribution:
        topic_probs[topic_id] = prob
    return topic_probs

# OCRからテキスト抽出
def extract_text_from_video(video_path, frame_interval=30):
    print(f"Extracting text from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    text_data = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            d = pytesseract.image_to_data(binary, output_type=Output.DICT, lang='jpn')

            for i in range(len(d['level'])):
                text = d['text'][i]
                if text.strip():
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    color = frame[y:y+h, x:x+w].mean(axis=(0, 1)).tolist()
                    text_data.append({
                        'text': text,
                        'position': (x, y),
                        'size': (w, h),
                        'color': color
                    })
        frame_number += 1

    cap.release()
    return text_data

# 感情分析
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 特徴量抽出
def extract_features(video_path, frame_count, dictionary, lda_model):
    print(f"Extracting text from video: {video_path}")
    text_data = extract_text_from_video(video_path)
    num_texts = len(text_data)
    avg_size = float(np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0)
    avg_color = [float(c) for c in (np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0])]

    combined_text = ' '.join([t['text'] for t in text_data])
    keywords = preprocess_text(combined_text)
    sentiment = float(analyze_sentiment(combined_text) if combined_text else 0)
    topic_distribution = get_topic_distribution(combined_text, dictionary, lda_model) if combined_text else [0]*10

    # video_id をファイル名から取得（整数への変換を行わない）
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

print("Preparing topic modeling...")
# トピックモデリングの準備
all_texts = []
for index, row in video_data.iterrows():
    text_data = extract_text_from_video(row['video_path'])
    combined_text = ' '.join([t['text'] for t in text_data])
    all_texts.append(combined_text)

tokenized_texts = [preprocess_text(text) for text in all_texts]
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

print("Extracting features from videos and saving to database...")
# 特徴量抽出とデータベースへの保存
video_features = video_data.apply(
    lambda row: extract_features(row['video_path'], row['frame_count'], dictionary, lda_model),
    axis=1
)
video_features_df = pd.DataFrame(video_features.tolist())

# --- ターゲット変数の取得 ---

# 必要なカラムを選択（例として '実績再生回数' を使用）
target_column_name = '実績再生回数'

print("\nRetrieving target variable from 'numeric_data' table...")
# 'numeric_data' テーブルから 'id' とターゲット変数を取得
target_data = numeric_data[['id', target_column_name]].copy()
target_data.rename(columns={'id': 'video_id', target_column_name: 'target'}, inplace=True)

# データ型の統一
# video_id を文字列型に変換
video_features_df['video_id'] = video_features_df['video_id'].astype(str)
target_data['video_id'] = target_data['video_id'].astype(str)

# 特徴量データとターゲットデータをマージ
data = pd.merge(video_features_df, target_data, on='video_id')

# データの欠損値を確認し、必要に応じて処理
print(f"Number of samples before dropping NA: {len(data)}")
data.dropna(subset=['target'], inplace=True)
print(f"Number of samples after dropping NA: {len(data)}")

# 特徴量とターゲットに分割
feature_columns = [
    'num_texts', 'avg_size', 'avg_color_r', 'avg_color_g', 'avg_color_b',
    'frame_count', 'sentiment',
    'topic_0', 'topic_1', 'topic_2', 'topic_3',
    'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9'
]

X = data[feature_columns]
y = data['target']

# データ型の変換（必要に応じて）
X = X.astype(float)
y = y.astype(float)

# --- モデルのトレーニングと評価 ---

print("\nTraining models...")

# ランダムフォレストモデル
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# 勾配ブースティングモデル
gb_model = GradientBoostingRegressor()
gb_model.fit(X, y)

# XGBoostモデル
xgb_model = XGBRegressor()
xgb_model.fit(X, y)

print("Models trained.")

# モデルの評価
print("\nEvaluating models...")

# 予測値の取得
y_pred_rf = rf_model.predict(X)
y_pred_gb = gb_model.predict(X)
y_pred_xgb = xgb_model.predict(X)

# RMSEの計算
rmse_rf = mean_squared_error(y, y_pred_rf, squared=False)
rmse_gb = mean_squared_error(y, y_pred_gb, squared=False)
rmse_xgb = mean_squared_error(y, y_pred_xgb, squared=False)

# R²スコアの計算
r2_rf = r2_score(y, y_pred_rf)
r2_gb = r2_score(y, y_pred_gb)
r2_xgb = r2_score(y, y_pred_xgb)

print(f"Random Forest RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")
print(f"Gradient Boosting RMSE: {rmse_gb:.2f}, R²: {r2_gb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}")

# --- モデルの保存 ---

print("\nSaving models...")
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(gb_model, 'gb_model.joblib')
joblib.dump(xgb_model, 'xgb_model.joblib')
print("Models saved.")

# --- 辞書とLDAモデルの保存 ---
print("\nSaving dictionary and LDA model...")
dictionary.save('dictionary.gensim')
lda_model.save('lda_model.gensim')
print("Dictionary and LDA model saved.")


# video_features テーブルの存在確認
cursor.execute("SHOW TABLES LIKE 'video_features'")
table_exists = cursor.fetchone()

# video_features テーブルの作成 (存在しない場合)
if not table_exists:
    cursor.execute('''
        CREATE TABLE video_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_id VARCHAR(255),
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
            topic_9 FLOAT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
    ''')
else:
    print("Adding topic columns to existing video_features table if not exists...")
    existing_columns_query = "SHOW COLUMNS FROM video_features"
    cursor.execute(existing_columns_query)
    existing_columns = [column[0] for column in cursor.fetchall()]

    topic_columns = [f"topic_{i}" for i in range(10)]
    for col in topic_columns:
        if col not in existing_columns:
            alter_table_query = f"ALTER TABLE video_features ADD COLUMN {col} FLOAT"
            cursor.execute(alter_table_query)

# 動画の特徴量を抽出し、データベースに保存 (上書き)
for index, row in video_features_df.iterrows():
    video_id = row['video_id']
    # 特徴量を辞書から取得
    features = row.to_dict()

    # SQLクエリを実行し、video_featuresテーブルに情報を上書き保存
    insert_query = """
    REPLACE INTO video_features (
        video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b,
        keywords, sentiment, topic_0, topic_1, topic_2, topic_3, topic_4,
        topic_5, topic_6, topic_7, topic_8, topic_9
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (
        video_id, features['num_texts'], features['avg_size'], features['avg_color_r'],
        features['avg_color_g'], features['avg_color_b'], features['keywords'], features['sentiment'],
        features['topic_0'], features['topic_1'], features['topic_2'], features['topic_3'],
        features['topic_4'], features['topic_5'], features['topic_6'], features['topic_7'],
        features['topic_8'], features['topic_9']
    ))
    conn.commit()

    print(f"Processed video ID: {video_id}, features inserted into database.")

# データベース接続を閉じる
print("Closing database connection...")
conn.close()

# video_featuresをCSVファイルに保存
print("Saving video features to CSV file...")
video_features_df.to_csv('video_features.csv', index=False)
print("Feature extraction and saving completed.")



# import os
# import cv2
# import mysql.connector as mydb
# import pandas as pd
# import numpy as np
# import librosa
# import moviepy.editor as mp
# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from textblob import TextBlob
# from gensim import corpora
# from gensim.models import LdaModel
# import spacy
# import configparser
# import pytesseract
# from pytesseract import Output
# import speech_recognition as sr

# # buzzAI.py から変数をインポート
# from buzzAI import numeric_columns, text_columns, date_columns

# print("Loading configuration file...")
# # 設定ファイルの読み込み
# config = configparser.ConfigParser()
# config.read('/Users/p10475/BuzzCity/config.ini')

# print("Connecting to the database...")
# # データベース接続
# conn = mydb.connect(
#     host=config['database']['host'],
#     port=config['database']['port'],
#     user=config['database']['user'],
#     password=config['database']['password'],
#     database=config['database']['database']
# )
# cursor = conn.cursor()

# print("Loading video data from the database...")
# # データベースから動画情報を読み込み
# cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
# video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

# print("Loading NLP model...")
# # NLPモデルの読み込み
# nlp = spacy.load("ja_core_news_sm")

# # テキストの前処理
# def preprocess_text(text, chunk_size=45000):
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#     tokens = []
#     for chunk in chunks:
#         doc = nlp(chunk)
#         tokens.extend([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
#     return tokens

# # MP4ファイルから音声を抽出し、librosaで特徴量を抽出
# def extract_audio_emotion_features_from_video(video_path):
#     print(f"Extracting audio emotion features from: {video_path}")
#     try:
#         # MoviePyでMP4からオーディオを抽出
#         video = mp.VideoFileClip(video_path)
#         audio = video.audio

#         # 一時的にWAVファイルとして保存
#         audio_path = "/tmp/temp_audio.wav"
#         audio.write_audiofile(audio_path)

#         # librosaでオーディオを読み込み、特徴量を抽出
#         y, sr = librosa.load(audio_path)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         mean_mfcc = np.mean(mfcc, axis=1)

#         # 一時ファイルの削除
#         os.remove(audio_path)
#         return mean_mfcc
#     except Exception as e:
#         print(f"Error processing {video_path}: {e}")
#         return np.zeros(13)  # エラー時にゼロの特徴量を返す

# # 音声からテキストに変換
# def convert_audio_to_text_from_video(video_path):
#     print(f"Converting audio to text from video: {video_path}")
#     try:
#         # MoviePyでMP4からオーディオを抽出
#         video = mp.VideoFileClip(video_path)
#         audio = video.audio

#         # 一時的にWAVファイルとして保存
#         audio_path = "/tmp/temp_audio.wav"
#         audio.write_audiofile(audio_path)

#         # SpeechRecognitionで音声をテキストに変換
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(audio_path) as source:
#             audio_data = recognizer.record(source)
#         text = recognizer.recognize_google(audio_data, language="ja-JP")

#         # 一時ファイルの削除
#         os.remove(audio_path)
#         return text
#     except Exception as e:
#         print(f"Error processing audio from {video_path}: {e}")
#         return ""

# # テキストの感情分析
# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# # OCRからテキスト抽出
# def extract_text_from_video(video_path):
#     print(f"Extracting text from video: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     text_data = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#         d = pytesseract.image_to_data(binary, output_type=Output.DICT)

#         for i in range(len(d['level'])):
#             text = d['text'][i]
#             if text.strip():
#                 x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
#                 color = frame[y:y+h, x:x+w].mean(axis=(0, 1)).tolist()
#                 text_data.append({
#                     'text': text,
#                     'position': (x, y),
#                     'size': (w, h),
#                     'color': color
#                 })

#     cap.release()
#     return text_data

# def get_topic_distribution(text, dictionary, lda_model):
#     tokens = preprocess_text(text)  # preprocess_text関数を使ってテキストをトークン化
#     bow = dictionary.doc2bow(tokens)  # テキストをBag of Words形式に変換
#     topic_distribution = lda_model.get_document_topics(bow)  # LDAモデルを使ってトピック分布を取得
#     return [topic_prob for topic_id, topic_prob in topic_distribution]

# # 特徴量抽出
# def extract_features(video_path, frame_count, dictionary, lda_model):
#     print(f"Extracting features from video: {video_path}")
#     text_data = extract_text_from_video(video_path)
#     num_texts = len(text_data)
#     avg_size = float(np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0)
#     avg_color = [float(c) for c in (np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0])]

#     combined_text = ' '.join([t['text'] for t in text_data])
#     keywords = preprocess_text(combined_text)
#     sentiment = float(analyze_sentiment(combined_text))
#     topic_distribution = get_topic_distribution(combined_text, dictionary, lda_model)

#     # 音声特徴量の抽出
#     audio_emotion_features = extract_audio_emotion_features_from_video(video_path)
#     audio_text = convert_audio_to_text_from_video(video_path)
#     audio_sentiment = analyze_sentiment(audio_text)

#     features = {
#         'num_texts': num_texts,
#         'avg_size': avg_size,
#         'avg_color_r': avg_color[0],
#         'avg_color_g': avg_color[1],
#         'avg_color_b': avg_color[2],
#         'frame_count': frame_count,
#         'keywords': ' '.join(keywords),
#         'sentiment': sentiment,
#         'audio_sentiment': audio_sentiment,
#         **{f'topic_{i}': float(topic_distribution[i]) if i < len(topic_distribution) else 0 for i in range(10)},
#         **{f'audio_feature_{i}': float(audio_emotion_features[i]) for i in range(len(audio_emotion_features))}
#     }

#     return features

# print("Preparing topic modeling...")
# # トピックモデリングの準備
# all_texts = []
# for index, row in video_data.iterrows():
#     text_data = extract_text_from_video(row['video_path'])
#     combined_text = ' '.join([t['text'] for t in text_data])
#     all_texts.append(combined_text)

# tokenized_texts = [preprocess_text(text) for text in all_texts]
# dictionary = corpora.Dictionary(tokenized_texts)
# corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
# lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# print("Extracting features from videos and saving to database...")
# # 特徴量抽出とデータベースへの保存
# video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count'], dictionary, lda_model), axis=1)
# video_features_df = pd.DataFrame(video_features.tolist())

# # video_features テーブルの存在確認
# cursor.execute("SHOW TABLES LIKE 'video_features'")
# table_exists = cursor.fetchone()

# # video_features テーブルの作成 (存在しない場合)
# if not table_exists:
#     cursor.execute('''
#         CREATE TABLE video_features (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             video_id BIGINT,
#             num_texts INT,
#             avg_size FLOAT,
#             avg_color_r FLOAT,
#             avg_color_g FLOAT,
#             avg_color_b FLOAT,
#             keywords TEXT,
#             sentiment FLOAT,
#             audio_sentiment FLOAT,
#             topic_0 FLOAT,
#             topic_1 FLOAT,
#             topic_2 FLOAT,
#             topic_3 FLOAT,
#             topic_4 FLOAT,
#             topic_5 FLOAT,
#             topic_6 FLOAT,
#             topic_7 FLOAT,
#             topic_8 FLOAT,
#             topic_9 FLOAT,
#             audio_feature_0 FLOAT,
#             audio_feature_1 FLOAT,
#             audio_feature_2 FLOAT,
#             audio_feature_3 FLOAT,
#             audio_feature_4 FLOAT,
#             audio_feature_5 FLOAT,
#             audio_feature_6 FLOAT,
#             audio_feature_7 FLOAT,
#             audio_feature_8 FLOAT,
#             audio_feature_9 FLOAT,
#             FOREIGN KEY (video_id) REFERENCES videos(video_id)
#         )
#     ''')
# else:
#     print("Adding columns to existing video_features table if not exists...")
#     existing_columns_query = "SHOW COLUMNS FROM video_features"
#     cursor.execute(existing_columns_query)
#     existing_columns = [column[0] for column in cursor.fetchall()]

#     new_columns = ['audio_sentiment'] + [f'audio_feature_{i}' for i in range(10)]
#     for col in new_columns:
#         if col not in existing_columns:
#             alter_table_query = f"ALTER TABLE video_features ADD COLUMN {col} FLOAT"
#             cursor.execute(alter_table_query)

# # 動画の特徴量を抽出し、データベースに保存 (上書き)
# for index, row in video_data.iterrows():
#     video_id = row['video_id']
#     video_path = row['video_path']
#     frame_count = row['frame_count']

#     # 特徴量抽出
#     features = extract_features(video_path, frame_count, dictionary, lda_model)

#     # SQLクエリを実行し、video_featuresテーブルに情報を上書き保存
#     insert_query = """
#     REPLACE INTO video_features (
#     video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b, 
#     keywords, sentiment, audio_sentiment, topic_0, topic_1, topic_2, 
#     topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9,
#     audio_feature_0, audio_feature_1, audio_feature_2, audio_feature_3, 
#     audio_feature_4, audio_feature_5, audio_feature_6, audio_feature_7, 
#     audio_feature_8, audio_feature_9, audio_feature_10, audio_feature_11, 
#     audio_feature_12
#     )
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """
#     print("Debugging SQL parameters:")
#     print(f"video_id: {video_id}")
#     print(f"features: {features}")
#     cursor.execute(insert_query, (
#         video_id, features['num_texts'], features['avg_size'], features['avg_color_r'], features['avg_color_g'], 
#         features['avg_color_b'], features['keywords'], features['sentiment'], features['audio_sentiment'], 
#         features['topic_0'], features['topic_1'], features['topic_2'], features['topic_3'], features['topic_4'], 
#         features['topic_5'], features['topic_6'], features['topic_7'], features['topic_8'], features['topic_9'],
#         features['audio_feature_0'], features['audio_feature_1'], features['audio_feature_2'], features['audio_feature_3'], 
#         features['audio_feature_4'], features['audio_feature_5'], features['audio_feature_6'], features['audio_feature_7'], 
#         features['audio_feature_8'], features['audio_feature_9'], features['audio_feature_10'], features['audio_feature_11'], 
#         features['audio_feature_12']
#     ))
#     conn.commit()

#     print(f"Processed video: {video_path}, features inserted into database.")

# # データベース接続を閉じる
# print("Closing database connection...")
# conn.close()

# # video_featuresをCSVファイルに保存
# print("Saving video features to CSV file...")
# video_features_df.to_csv('video_features.csv', index=False)
# print("Feature extraction and saving completed.")


