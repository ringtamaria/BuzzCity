import os
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import configparser
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

# buzzAI.py から変数をインポート
from buzzAI import numeric_columns, text_columns, date_columns

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

# データベースから数値データを読み込み
cursor.execute("SELECT * FROM numeric_data")
numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースからテキストデータを読み込み
cursor.execute("SELECT * FROM text_data")
text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースから日付データを読み込み
cursor.execute("SELECT * FROM date_data")
date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースから動画特徴量データを読み込み
cursor.execute("SELECT * FROM video_features")
video_features_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                          np.where(df[column] < lower_bound, lower_bound, df[column]))

# データ前処理
for col in numeric_columns:
    handle_outliers(numeric_data, col)

# スケーリング
scaler = StandardScaler()
numeric_data[numeric_columns] = scaler.fit_transform(numeric_data[numeric_columns])

# 数値データの正規化
pt = PowerTransformer(method='box-cox')
for col in numeric_columns:
    if (numeric_data[col] > 0).all():  # Box-Cox変換は正の値のみを対象とする
        numeric_data[col] = pt.fit_transform(numeric_data[[col]])

# テキストデータのエンコーディング
encoder = TargetEncoder()
for col in text_columns:
    if col in text_data.columns:
        text_data[f'encoded_{col}'] = encoder.fit_transform(text_data[col], numeric_data['動画視聴数'])

# テキストカラムの削除（エンコード後は元のテキストカラムは不要）
text_data = text_data.drop(columns=text_columns, errors='ignore')

# 動画データがあるもの且つ動画視聴数が０ではないものをフィルタリング
valid_videos = video_data[video_data['video_id'].isin(numeric_data[numeric_data['動画視聴数'] != 0]['id'])]

# 有効な動画の数をカウント
valid_video_count = valid_videos.shape[0]
print(f"Number of valid videos used for training: {valid_video_count}")

# データを結合
merged_data = pd.merge(valid_videos[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')

# 日付データをエポック時間に変換
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col])
    merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# 特徴量とターゲットの分割
X = merged_data.drop(['video_id', '動画視聴数'], axis=1)  # video_idと動画視聴数は学習には不要
y = merged_data['動画視聴数']  # ターゲット変数

# 特徴量選択
selector = SelectKBest(f_regression, k='all')  # k='all'は全ての特徴量を使うことを意味しますが、必要に応じて数を調整してください
X_selected = selector.fit_transform(X, y)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# モデル選択と学習
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'SVM': SVR(),
    'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,)),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor()
}

param_grids = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'learning_rate_init': [0.001, 0.01]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    y_pred = best_models[model_name].predict(X_test)
    print(f"{model_name} - MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}, MAE: {mean_absolute_error(y_test, y_pred)}")

# 結果のプロット
plt.figure(figsize=(10, 5))
for model_name, model in best_models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    plt.plot(train_sizes, -train_scores.mean(axis=1), label=f'{model_name} Train')
    plt.plot(train_sizes, -test_scores.mean(axis=1), label=f'{model_name} Test')

plt.xlabel('Training examples')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()




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
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

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
#     keywords = extract_keywords(combined_text)
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
# video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count']), axis=1)
# video_features_df = pd.DataFrame(video_features.tolist())

# # video_featuresテーブルにデータを挿入
# for index, row in video_features_df.iterrows():
#     insert_query = """
#     INSERT INTO video_features (video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b, keywords, sentiment)
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (row['video_id'], row['num_texts'], row['avg_size'], row['avg_color_r'], row['avg_color_g'], row['avg_color_b'], row['keywords'], row['sentiment']))
# conn.commit()

# # 動画データがあるもの且つ動画視聴数が０ではないものをフィルタリング
# valid_videos = video_data[video_data['video_id'].isin(numeric_data[numeric_data['動画視聴数'] != 0]['id'])]

# # 有効な動画の数をカウント
# valid_video_count = valid_videos.shape[0]
# print(f"Number of valid videos used for training: {valid_video_count}")

# # データを結合
# merged_data = pd.merge(valid_videos[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
# merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')

# # 日付データをエポック時間に変換
# for col in date_columns:
#     merged_data[col] = pd.to_datetime(merged_data[col])
#     merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# # 特徴量とターゲットの分割
# X = merged_data.drop(['video_id', '動画視聴数'], axis=1)  # video_idと動画視聴数は学習には不要
# y = merged_data['動画視聴数']  # ターゲット変数

# # 特徴量選択
# selector = SelectKBest(f_regression, k='all')  # k='all'は全ての特徴量を使うことを意味しますが、必要に応じて数を調整してください
# X_selected = selector.fit_transform(X, y)

# # データ分割
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# # モデル選択と学習
# models = {
#     'RandomForest': RandomForestRegressor(),
#     'GradientBoosting': GradientBoostingRegressor(),
#     'SVM': SVR(),
#     'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,)),
#     'XGBoost': XGBRegressor(),
#     'LightGBM': LGBMRegressor()
# }

# param_grids = {
#     'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]},
#     'GradientBoosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]},
#     'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
#     'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'alpha': [0.0001, 0.001]},
#     'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]},
#     'LightGBM': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
# }

# best_models = {}
# for name, model in models.items():
#     grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
#     best_models[name] = grid_search.best_estimator_

# # アンサンブル学習
# voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
# voting_regressor.fit(X_train, y_train)

# # 学習曲線のプロット
# train_sizes, train_scores, test_scores = learning_curve(voting_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

# train_scores_mean = -train_scores.mean(axis=1)
# test_scores_mean = -test_scores.mean(axis=1)

# plt.figure()
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title('Learning Curve')
# plt.xlabel('Training Size')
# plt.ylabel('MSE')
# plt.legend(loc="best")
# plt.show()

# # モデルの評価
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
#     print(f'Model: {model.__class__.__name__}')
#     print(f'Mean Squared Error: {mse}')
#     print(f'R^2 Score: {r2}')
#     print(f'Mean Absolute Error: {mae}')
#     print(f'Mean Absolute Percentage Error: {mape}')
    
#     # 残差プロット
#     residuals = y_test - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_pred, residuals)
#     plt.axhline(0, color='red', linestyle='--')
#     plt.xlabel('Predicted Values')
#     plt.ylabel('Residuals')
#     plt.title(f'Residuals vs Predicted Values for {model.__class__.__name__}')
#     plt.show()

# # 各モデルの評価
# for name, model in best_models.items():
#     evaluate_model(model, X_test, y_test)

# # アンサンブルモデルの評価
# evaluate_model(voting_regressor, X_test, y_test)

# # データベース接続を閉じる
# conn.close()