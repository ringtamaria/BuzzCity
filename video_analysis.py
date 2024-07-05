import os
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import configparser
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt  # matplotlib をインポート

# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read('/Users/p10475/BuzzCity/config.ini')

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

# OCRの準備 (Tesseract OCRのパスを設定)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

# OCRでテロップ情報を抽出 (フレームのサンプリングを追加)
def extract_text_from_video(video_path, sampling_rate=1):  # sampling_rate を引数に追加
    cap = cv2.VideoCapture(video_path)
    text_data = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in range(0, frame_count, sampling_rate):  # sampling_rate ごとに処理
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # テロップ抽出処理 (例: 色やサイズでフィルタリング)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        d = pytesseract.image_to_data(binary, output_type=Output.DICT)

        for i in range(len(d['level'])):
            text = d['text'][i]
            if text.strip():  # 空白文字列は除外
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                color = frame[y:y+h, x:x+w].mean(axis=(0,1)).tolist()
                text_data.append({
                    'text': text,
                    'position': (x, y),
                    'size': (w, h),
                    'color': color
                })

    cap.release()
    return text_data

# 特徴量生成
def extract_features(video_path, frame_count):
    text_data = extract_text_from_video(video_path)
    
    # テロップの数、平均サイズ、平均色などの特徴量を計算
    num_texts = len(text_data)
    avg_size = np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0
    avg_color = np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0]
    
    features = {
        'num_texts': num_texts,
        'avg_size': avg_size,
        'avg_color_r': avg_color[0],
        'avg_color_g': avg_color[1],
        'avg_color_b': avg_color[2],
        'frame_count': frame_count
    }

    return features

video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count']), axis=1)
video_features_df = pd.DataFrame(video_features.tolist())

# データ結合
merged_data = pd.merge(video_data[['video_id', 'frame_count']], csv_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.concat([merged_data, video_features_df], axis=1)

# データの前処理
# ... (必要に応じて欠損値処理や特徴量エンジニアリングなどを行う)

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

# データベース接続を閉じる
conn.close()


# import os
# import cv2
# import mysql.connector as mydb
# import pandas as pd
# import numpy as np
# import openai  # openaiライブラリをインポート
# import configparser
# from sklearn.cluster import KMeans
# import pytesseract
# from pytesseract import Output
# import time
# import matplotlib.pyplot as plt  # matplotlib をインポート

# # 設定ファイルの読み込み
# config = configparser.ConfigParser()
# config.read('/Users/p10475/BuzzCity/config.ini')

# # OpenAI APIキーを設定
# openai.api_key = config['openai']['api_key']

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

# # データベースからCSVデータを読み込み
# cursor.execute("SELECT * FROM numeric_data")
# csv_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

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
#                 color = frame[y:y+h, x:x+w].mean(axis=(0,1)).tolist()
#                 text_data.append({
#                     'text': text,
#                     'position': (x, y),
#                     'size': (w, h),
#                     'color': color
#                 })

#     cap.release()
#     return text_data

# # GPT-4で動画内容を分析
# def analyze_video_with_gpt4(video_path, retry_count=3, retry_delay=5):
#     for i in range(retry_count):
#         try:
#             text_data = extract_text_from_video(video_path)
#             chunk_size = 100  # メッセージのチャンクサイズを設定
#             responses = []

#             for i in range(0, len(text_data), chunk_size):
#                 chunk = text_data[i:i+chunk_size]
#                 response = openai.ChatCompletion.create(  # openai.ChatCompletion.create を使用する
#                     model="gpt-4",
#                     messages=[
#                         {"role": "system", "content": "あなたは動画分析AIです。"},
#                         {"role": "user", "content": f"以下の動画のテロップ情報を分析してください。\n{chunk}"}
#                     ]
#                 )
#                 responses.append(response.choices[0].message["content"].strip())

#             return " ".join(responses), True  # 分析結果とAPI使用フラグを返す

#         except openai.error.OpenAIError:  # openai.error.OpenAIError を使用する
#             if i < retry_count - 1:  # 最終試行でなければ
#                 print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
#                 time.sleep(retry_delay)
#             else:
#                 print("Rate limit exceeded. Skipping GPT-4 analysis.")
#                 return None, False  # NoneとAPI未使用フラグを返す

# # 特徴量生成
# def extract_features(video_path, frame_count):
#     analysis_result, used_gpt4 = analyze_video_with_gpt4(video_path)

#     if used_gpt4:
#         # GPT-4の分析結果から特徴量を抽出
#         features = {
#             'sentiment': get_sentiment(analysis_result),
#             'themes': get_themes(analysis_result),
#             'target_audience': get_target_audience(analysis_result),
#             'frame_count': frame_count
#         }
#     else:
#         # GPT-4を使わない場合の特徴量を抽出
#         features = {
#             'frame_count': frame_count
#         }

#     return features


# # 特徴量抽出関数の例 (仮実装)
# def get_sentiment(text):
#     # 感情分析APIなどを使って感情を判定
#     return "ポジティブ"  # 仮の値

# def get_themes(text):
#     # テーマ抽出処理
#     return ["テーマ1", "テーマ2"]  # 仮の値

# def get_target_audience(text):
#     # 視聴者層分析処理
#     return "10代女性"  # 仮の値

# video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count']), axis=1)
# video_features_df = pd.DataFrame(video_features.tolist())

# # データ結合
# merged_data = pd.merge(video_data[['video_id', 'frame_count']], csv_data, left_on='video_id', right_on='id', how='inner')
# merged_data = pd.concat([merged_data, video_features_df], axis=1)

# # データの前処理
# # ... (必要に応じて欠損値処理や特徴量エンジニアリングなどを行う)

# # 特徴量を数値データに変換（例：One-Hot Encoding）
# # ...

# # 教師なし学習 (例: K-meansクラスタリング)
# X = merged_data.drop(['video_id', 'video_path', 'id'], axis=1)  # video_idとvideo_pathは学習には不要
# kmeans = KMeans(n_clusters=5, random_state=0)  # 5つのクラスタに分類
# cluster_labels = kmeans.fit_predict(X)

# # 可視化
# plt.figure(figsize=(10, 6))
# plt.scatter(merged_data['実績再生回数'], merged_data['frame_count'], c=cluster_labels, cmap='viridis')
# plt.xlabel('実績再生回数')
# plt.ylabel('フレーム数')
# plt.title('TikTok動画クラスタリング結果')

# # 各動画の情報をグラフ上に表示
# for i, row in merged_data.iterrows():
#     plt.annotate(f"{row['video_id']}", (row['実績再生回数'], row['frame_count']), textcoords="offset points", xytext=(0,10), ha='center')

# plt.show()

# # データベース接続を閉じる
# conn.close()
