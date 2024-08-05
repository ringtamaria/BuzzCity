import os
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
import spacy
import configparser
import pytesseract
from pytesseract import Output

# buzzAI.py から変数をインポート
from buzzAI import numeric_columns, text_columns, date_columns

print("Loading configuration file...")
# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read('/Users/p10475/BuzzCity/config.ini')

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
    return [topic_prob for topic_id, topic_prob in topic_distribution]

# OCRからテキスト抽出
def extract_text_from_video(video_path):
    print(f"Extracting text from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    text_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        d = pytesseract.image_to_data(binary, output_type=Output.DICT)

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
    sentiment = float(analyze_sentiment(combined_text))
    topic_distribution = get_topic_distribution(combined_text, dictionary, lda_model)

    features = {
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
print("Preparing topic modeling...")
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
video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count'], dictionary, lda_model), axis=1)
video_features_df = pd.DataFrame(video_features.tolist())

# video_features テーブルの存在確認
cursor.execute("SHOW TABLES LIKE 'video_features'")
table_exists = cursor.fetchone()

# video_features テーブルの作成 (存在しない場合)
if not table_exists:
    cursor.execute('''
        CREATE TABLE video_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_id BIGINT,
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
for index, row in video_data.iterrows():
    video_id = row['video_id']
    video_path = row['video_path']
    frame_count = row['frame_count']

    # 特徴量抽出
    features = extract_features(video_path, frame_count, dictionary, lda_model)

    # SQLクエリを実行し、video_featuresテーブルに情報を上書き保存
    insert_query = """
    REPLACE INTO video_features (video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b, keywords, sentiment, topic_0, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (
        video_id, features['num_texts'], features['avg_size'], features['avg_color_r'], features['avg_color_g'], features['avg_color_b'],
        features['keywords'], features['sentiment'], features['topic_0'], features['topic_1'], features['topic_2'], features['topic_3'],
        features['topic_4'], features['topic_5'], features['topic_6'], features['topic_7'], features['topic_8'], features['topic_9']
    ))
    conn.commit()

    print(f"Processed video: {video_path}, features inserted into database.")

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
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import configparser
# import pytesseract
# from pytesseract import Output
# import matplotlib.pyplot as plt
# import seaborn as sns
# import spacy
# from textblob import TextBlob
# from category_encoders import TargetEncoder
# from sklearn.feature_selection import SelectFromModel

# # buzzAI.py から変数をインポート
# from buzzAI import numeric_columns, text_columns, date_columns

# # 設定ファイルの読み込み
# config = configparser.ConfigParser()
# config.read('/Users/p10475/BuzzCity/config.ini')

# # データベース接続
# conn = mydb.connect(
#     host=config['database']['host'],
#     port=config['database']['port'],
#     user=config['database']['user'],
#     password=config['database']['password'],
#     database=config['database']['database']
# )
# cursor = conn.cursor()

# # データベースから動画情報を読み込み
# cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
# video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

# # データベースから数値データを読み込み
# cursor.execute("SELECT * FROM numeric_data")
# numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # データベースからテキストデータを読み込み
# cursor.execute("SELECT * FROM text_data")
# text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # データベースから日付データを読み込み
# cursor.execute("SELECT * FROM date_data")
# date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # OCRの準備 (Tesseract OCRのパスを設定)
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

# def extract_text_from_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     text_data = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # テロップ抽出処理 (例: 色やサイズでフィルタリング)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#         d = pytesseract.image_to_data(binary, output_type=Output.DICT)

#         for i in range(len(d['level'])):
#             text = d['text'][i]
#             if text.strip():  # 空白文字列は除外
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

# def handle_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df[column] = np.where(df[column] > upper_bound, upper_bound,
#                           np.where(df[column] < lower_bound, lower_bound, df[column]))

# nlp = spacy.load("ja_core_news_sm")

# def extract_keywords(text, chunk_size=45000):  # chunk_sizeで分割サイズを指定
#     keywords = []
#     for i in range(0, len(text), chunk_size):
#         chunk = text[i:i+chunk_size]
#         doc = nlp(chunk)
#         keywords.extend([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
#     return keywords

# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# # 特徴量生成
# def extract_features(video_path, frame_count):
#     text_data = extract_text_from_video(video_path)

#     # テロップの数、平均サイズ、平均色などの特徴量を計算
#     num_texts = len(text_data)
#     avg_size = np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0
#     avg_color = np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0]

#     combined_text = ' '.join([t['text'] for t in text_data])
    
#     # テキストを適切な長さに分割し、キーワードを抽出
#     keywords = extract_keywords(combined_text, chunk_size=45000)
#     sentiment = analyze_sentiment(combined_text)

#     features = {
#         'num_texts': num_texts,
#         'avg_size': float(avg_size),      # float64 -> float に変換
#         'avg_color_r': float(avg_color[0]),  # float64 -> float に変換
#         'avg_color_g': float(avg_color[1]),  # float64 -> float に変換
#         'avg_color_b': float(avg_color[2]),  # float64 -> float に変換
#         'frame_count': frame_count,
#         'keywords': ' '.join(keywords),
#         'sentiment': float(sentiment)     # float64 -> float に変換
#     }

#     return features

# # データ前処理
# for col in numeric_columns:
#     handle_outliers(numeric_data, col)

# # スケーリング
# scaler = StandardScaler()
# numeric_data[numeric_columns] = scaler.fit_transform(numeric_data[numeric_columns])

# # 数値データの正規化
# pt = PowerTransformer(method='box-cox')
# for col in numeric_columns:
#     if (numeric_data[col] > 0).all():  # Box-Cox変換は正の値のみを対象とする
#         numeric_data[col] = pt.fit_transform(numeric_data[[col]])

# # テキストデータのエンコーディング
# encoder = TargetEncoder()
# for col in text_columns:
#     if col in text_data.columns:
#         text_data[f'encoded_{col}'] = encoder.fit_transform(text_data[col], numeric_data['動画視聴数'])

# video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count']), axis=1)
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
#             FOREIGN KEY (video_id) REFERENCES videos(video_id)
#         )
#     ''')

# # 動画の特徴量を抽出し、データベースに保存
# for index, row in video_data.iterrows():
#     video_id = row['video_id']
#     video_path = row['video_path']
#     frame_count = row['frame_count']

#     # 特徴量抽出
#     features = extract_features(video_path, frame_count)

#     # SQLクエリを実行し、video_featuresテーブルに情報を挿入
#     insert_query = """
#     INSERT INTO video_features (video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b, keywords, sentiment)
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (video_id, features['num_texts'], features['avg_size'], features['avg_color_r'], features['avg_color_g'], features['avg_color_b'], features['keywords'], features['sentiment']))
#     conn.commit()

#     print(f"Processed video: {video_path}, features inserted into database.")

# # データベース接続を閉じる
# conn.close()


# # video_featuresをCSVファイルに保存
# video_features_df.to_csv('video_features.csv', index=False)