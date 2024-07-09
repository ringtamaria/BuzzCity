import os
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import configparser
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from textblob import TextBlob
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

# データベースからCSVデータを読み込み
cursor.execute("SELECT * FROM numeric_data")
csv_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# OCRの準備 (Tesseract OCRのパスを設定)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    text_data = []

    while True:
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
                color = frame[y:y+h, x:x+w].mean(axis=(0, 1)).tolist()
                text_data.append({
                    'text': text,
                    'position': (x, y),
                    'size': (w, h),
                    'color': color
                })

    cap.release()
    return text_data

def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                          np.where(df[column] < lower_bound, lower_bound, df[column]))

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 特徴量生成
def extract_features(video_path, frame_count):
    text_data = extract_text_from_video(video_path)

    # テロップの数、平均サイズ、平均色などの特徴量を計算
    num_texts = len(text_data)
    avg_size = np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0
    avg_color = np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0]

    combined_text = ' '.join([t['text'] for t in text_data])
    keywords = extract_keywords(combined_text)
    sentiment = analyze_sentiment(combined_text)

    features = {
        'num_texts': num_texts,
        'avg_size': avg_size,
        'avg_color_r': avg_color[0],
        'avg_color_g': avg_color[1],
        'avg_color_b': avg_color[2],
        'frame_count': frame_count,
        'keywords': ' '.join(keywords),
        'sentiment': sentiment
    }

    return features

# 手動でカラムを指定してデータを分割
numeric_columns = ['広告アカウントID',
                   'campaign_id',
                   '想定再生回数',
                   '実績再生回数',
                   '総額',
                   '配信予算',
                   '配信費用',
                   '想定再生単価',
                   '実績再生単価',
                   '視聴達成率',
                   '掲載単価N',  # カッコを削除
                   '動画秒数',
                   'いいね',
                   'いいね率',
                   'コメント',
                   'コメント率',
                   'シェア',
                   'シェア率',
                   '保存',
                   '保存率',
                   'ENG数',
                   'ENG率',
                   'cpc',
                   'CPM',
                   'IMP数',
                   'CL数',
                   'CTR',
                   'リーチ数',
                   'FQ',
                   'リーチ率',
                   '動画視聴数',
                   '2秒視聴率',
                   '2秒動画再生数',
                   '6秒視聴率',
                   '6秒動画再生数',
                   '完全視聴率',
                   '再生完了数',
                   '再生長さ75パーセント',  # カッコを削除し、表現を変更
                   '再生長さ50パーセント',  # カッコを削除し、表現を変更
                   '再生長さ25パーセント',  # カッコを削除し、表現を変更
                   '平均再生秒数',
                   '一人当たりの平均視聴時間',
                   'オーガニック再生数',  # カッコを削除
                   'リーチ単価',
                   '視聴単価',
                   '完全視聴単価',
                   'エンゲージ単価']
text_columns = ['企業名', 'クライアント名', '商品名', 'カテゴリー', 'URL', '配信ステータス', '配信目的']
date_columns = ['投稿日']

# データ前処理
for col in numeric_columns:
    handle_outliers(csv_data, col)

# スケーリング
scaler = StandardScaler()
csv_data[numeric_columns] = scaler.fit_transform(csv_data[numeric_columns])

# 数値データの正規化
pt = PowerTransformer(method='box-cox')
for col in numeric_columns:
    if (csv_data[col] > 0).all():  # Box-Cox変換は正の値のみを対象とする
        csv_data[col] = pt.fit_transform(csv_data[[col]])

# テキストデータのエンコーディング
encoder = TargetEncoder()
for col in text_columns:
    if col in csv_data.columns:
        csv_data[f'encoded_{col}'] = encoder.fit_transform(csv_data[col], csv_data['動画視聴数'])

video_features = video_data.apply(lambda row: extract_features(row['video_path'], row['frame_count']), axis=1)
video_features_df = pd.DataFrame(video_features.tolist())

# データ結合
merged_data = pd.merge(video_data[['video_id', 'frame_count']], csv_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.concat([merged_data, video_features_df], axis=1)

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
    'NeuralNetwork': MLPRegressor()
}

param_grids = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]},
    'GradientBoosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'alpha': [0.0001, 0.001]}
}

best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_

# アンサンブル学習
voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
voting_regressor.fit(X_train, y_train)

# 学習曲線のプロット
train_sizes, train_scores, test_scores = learning_curve(voting_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('MSE')
plt.legend(loc="best")
plt.show()

# モデルの評価
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f'Model: {model.__class__.__name__}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')
    
    # 残差プロット
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for {model.__class__.__name__}')
    plt.show()

# 各モデルの評価
for name, model in best_models.items():
    evaluate_model(model, X_test, y_test)

# アンサンブルモデルの評価
evaluate_model(voting_regressor, X_test, y_test)

# データベース接続を閉じる
conn.close()

