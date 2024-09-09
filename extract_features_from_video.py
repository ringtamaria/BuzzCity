import os
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import configparser
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# buzzAI.py から変数をインポート
from buzzAI import numeric_columns, text_columns, date_columns

def print_progress(message):
    print(f"[Progress] {message}")

# 設定ファイルの読み込み
print_progress("Loading configuration file...")
config = configparser.ConfigParser()
config.read('/Users/p10475/BuzzCity/config.ini')

# データベース接続
print_progress("Connecting to the database...")
conn = mydb.connect(
    host=config['database']['host'],
    port=config['database']['port'],
    user=config['database']['user'],
    password=config['database']['password'],
    database=config['database']['database']
)
cursor = conn.cursor()

# データベースから各種データを読み込み
print_progress("Loading data from the database...")
cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

cursor.execute("SELECT * FROM numeric_data")
numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

cursor.execute("SELECT * FROM text_data")
text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

cursor.execute("SELECT * FROM date_data")
date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

cursor.execute("SELECT * FROM video_features")
video_features_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# テキストデータのエンコーディング
print_progress("Encoding text data...")
encoder = TargetEncoder()
for col in text_columns:
    if col in text_data.columns:
        text_data[f'encoded_{col}'] = encoder.fit_transform(text_data[col], numeric_data['動画視聴数'])

# IDの追加
text_data['id'] = numeric_data['id']

# エンコーディング結果を保存するためのテーブルを作成
def create_encoded_text_table(conn):
    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS encoded_text_data (
        id INT PRIMARY KEY,
        {}
    )
    """.format(", ".join([f"encoded_{col} FLOAT" for col in text_columns]))
    cursor.execute(create_table_query)
    conn.commit()

# テーブルの作成
create_encoded_text_table(conn)

# エンコーディング結果をデータベースに保存
def save_encoded_text_to_db(conn, encoded_text_data):
    cursor = conn.cursor()
    for index, row in encoded_text_data.iterrows():
        values = [float(row[f'encoded_{col}']) for col in text_columns]
        insert_query = """
        REPLACE INTO encoded_text_data (id, {})
        VALUES (%s, {})
        """.format(
            ", ".join([f"encoded_{col}" for col in text_columns]),
            ", ".join(["%s"] * len(text_columns))
        )
        cursor.execute(insert_query, (int(row['id']), *values))
    conn.commit()

# エンコーディング結果の保存
print_progress("Saving encoded text data to the database...")
save_encoded_text_to_db(conn, text_data)

# データベースからエンコーディング結果を読み込む
def load_encoded_text_from_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM encoded_text_data")
    encoded_text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    return encoded_text_data

# エンコーディング結果を読み込んで結合
print_progress("Loading encoded text data from the database...")
encoded_text_data = load_encoded_text_from_db(conn)

# データを結合
print_progress("Checking for 'id' column in all dataframes...")
# IDカラムがあることを確認する
if 'id' not in numeric_data.columns or 'id' not in text_data.columns or 'id' not in date_data.columns:
    raise KeyError("ID column is missing in one of the datasets.")

print_progress("Merging data...")
merged_data = pd.merge(video_data[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')
merged_data = pd.merge(merged_data, encoded_text_data, on='id', how='inner')

# 日付データをエポック時間に変換
print_progress("Converting date columns to epoch time...")
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col])
    merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# 欠損値の補完
print_progress("Imputing missing values...")
numeric_features = merged_data.select_dtypes(include=[np.number])
categorical_features = merged_data.select_dtypes(exclude=[np.number])

imputer_numeric = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

numeric_features_imputed = imputer_numeric.fit_transform(numeric_features)
categorical_features_imputed = imputer_categorical.fit_transform(categorical_features)

# OneHotEncoderを使用して非数値データを数値データに変換
print_progress("Encoding categorical features...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical_features = encoder.fit_transform(categorical_features_imputed)

# 数値データとエンコードされた非数値データを再結合
print_progress("Combining features...")
X = np.hstack((numeric_features_imputed, encoded_categorical_features))

# ターゲット変数のスケーリング
print_progress("Scaling target variable...")
y = merged_data['動画視聴数'].values.reshape(-1, 1)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y).flatten()

# 特徴量選択
print_progress("Selecting features...")
selector = SelectKBest(f_regression, k='all')
X_selected = selector.fit_transform(X, y_scaled)

# モデル選択と学習
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

param_grids = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}  # Early stopping removed
}

best_models = {}
predictions = {}
for model_name, model in models.items():
    print_progress(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_selected, y_scaled)
    best_models[model_name] = grid_search.best_estimator_
    y_pred = best_models[model_name].predict(X_selected)
    
    # 予測値の逆スケーリングとマイナス値の処理
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)
    
    predictions[model_name] = y_pred
    
    print(f"{model_name} - MSE: {mean_squared_error(y_scaled, y_pred)}, R2: {r2_score(y_scaled, y_pred)}, MAE: {mean_absolute_error(y_scaled, y_pred)}")

# グラフの保存先ディレクトリ
result_dir = '/Users/p10475/BuzzCity/result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 全データに対する予測結果の保存
output_file_name = 'prediction_results_all_data.csv'
print_progress(f"Saving prediction results to {output_file_name}...")
prediction_results = pd.DataFrame({
    'Actual View Count': target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
})
for model_name, pred in predictions.items():
    prediction_results[f'{model_name} Predicted View Count'] = pred

prediction_results.to_csv(os.path.join(result_dir, output_file_name), index=False)

# 学習曲線のプロットと保存
print_progress("Plotting learning curves...")
plt.figure(figsize=(10, 5))
for model_name, model in best_models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, X_selected, y_scaled, cv=3, scoring='neg_mean_squared_error')
    plt.plot(train_sizes, -train_scores.mean(axis=1), label=f'{model_name} Train')
    plt.plot(train_sizes, -test_scores.mean(axis=1), label=f'{model_name} Test')

plt.xlabel('Training examples')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves for Different Models')
plt.legend()
plt.grid()
plt.savefig(os.path.join(result_dir, 'learning_curves.png'))
plt.show()

# 全データに対する予測結果を可視化
print_progress("Visualizing predictions vs actual view counts for all data...")
plt.figure(figsize=(12, 6))
x = np.arange(len(y_scaled))

for model_name, y_pred in predictions.items():
    plt.plot(x, y_pred, label=f'{model_name} Predictions')

plt.plot(x, target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten(), 'k--', label='Actual Values', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('View Count')
plt.title('Predicted vs Actual View Counts for All Data')
plt.legend()
plt.savefig(os.path.join(result_dir, 'predicted_vs_actual_all_data.png'))
plt.show()

# アンサンブル学習
print_progress("Training Voting Regressor...")
voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
voting_regressor.fit(X_selected, y_scaled)

# スタッキングアンサンブル学習
print_progress("Training Stacking Regressor...")
estimators = [('rf', RandomForestRegressor(**best_models['RandomForest'].get_params())),
              ('gb', GradientBoostingRegressor(**best_models['GradientBoosting'].get_params()))]
final_estimator = LinearRegression()
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
stacking_regressor.fit(X_selected, y_scaled)

# 学習曲線のプロットと保存
print_progress("Plotting learning curves for ensemble models...")
models_for_learning_curve = {
    'VotingRegressor': voting_regressor,
    'StackingRegressor': stacking_regressor
}

for name, model in models_for_learning_curve.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_selected, y_scaled, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f'Learning Curve for {name}')
    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.legend(loc="best")
    plt.savefig(os.path.join(result_dir, f'learning_curve_{name}_all_data.png'))
    plt.show()

# モデルの評価
def evaluate_model(model, X_data, y_data, model_name):
    print_progress(f"Evaluating {model_name}...")
    y_pred = model.predict(X_data)
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)
    y_data_original = target_scaler.inverse_transform(y_data.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_data_original, y_pred)
    r2 = r2_score(y_data_original, y_pred)
    mae = mean_absolute_error(y_data_original, y_pred)
    mape = np.mean(np.abs((y_data_original - y_pred) / y_data_original)) * 100 if not (y_data_original == 0).any() else np.inf
    
    print(f'Model: {model_name}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')
    
    # 残差プロット
    print_progress(f"Plotting residuals for {model_name}...")
    residuals = y_data_original - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for {model_name}')
    plt.savefig(os.path.join(result_dir, f'residuals_{model_name}_all_data.png'))
    plt.show()

# 各モデルの評価
for name, model in best_models.items():
    evaluate_model(model, X_selected, y_scaled, name)

# アンサンブルモデルの評価
evaluate_model(voting_regressor, X_selected, y_scaled, 'VotingRegressor')
evaluate_model(stacking_regressor, X_selected, y_scaled, 'StackingRegressor')
# データベース接続を閉じる
print_progress("Closing database connection...")
conn.close()
print_progress("Process completed.")



# import os
# import cv2
# import pandas as pd
# import numpy as np
# from gensim import corpora
# from gensim.models import LdaModel
# import spacy
# from pytesseract import Output
# import pytesseract
# import mysql.connector as mydb
# import configparser
# from video_analysis import extract_text_from_video, preprocess_text, analyze_sentiment, get_topic_distribution

# # NLPモデルの読み込み
# nlp = spacy.load("ja_core_news_sm")

# # 新しいテスト動画フォルダのパス
# test_video_folder = '/Users/p10475/BuzzCity/tiktok_testvideo'

# # データベース接続の設定
# def connect_to_database():
#     config = configparser.ConfigParser()
#     config.read('/Users/p10475/BuzzCity/config.ini')
#     conn = mydb.connect(
#         host=config['database']['host'],
#         port=config['database']['port'],
#         user=config['database']['user'],
#         password=config['database']['password'],
#         database=config['database']['database']
#     )
#     return conn

# # 動画のフレーム数を取得
# def get_frame_count(video_test_path):
#     cap = cv2.VideoCapture(video_test_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return frame_count

# # 動画の特徴量を抽出する関数
# def extract_features_from_video(video_test_path, frame_count, dictionary, lda_model):
#     print(f"Extracting features from video: {video_test_path}")
    
#     # テキストデータの抽出
#     text_data = extract_text_from_video(video_test_path)
#     num_texts = len(text_data)
#     avg_size = float(np.mean([t['size'][0] * t['size'][1] for t in text_data]) if text_data else 0)
#     avg_color = [float(c) for c in (np.mean([t['color'] for t in text_data], axis=0) if text_data else [0, 0, 0])]
    
#     combined_text = ' '.join([t['text'] for t in text_data])
#     keywords = preprocess_text(combined_text)
#     sentiment = float(analyze_sentiment(combined_text))
#     topic_distribution = get_topic_distribution(combined_text, dictionary, lda_model)

#     # 特徴量のディクショナリ
#     features = {
#         'num_texts': num_texts,
#         'avg_size': avg_size,
#         'avg_color_r': avg_color[0],
#         'avg_color_g': avg_color[1],
#         'avg_color_b': avg_color[2],
#         'frame_count': frame_count,
#         'keywords': ' '.join(keywords),
#         'sentiment': sentiment,
#         **{f'topic_{i}': float(topic_distribution[i]) if i < len(topic_distribution) else 0 for i in range(10)}
#     }

#     return features

# # 動画の特徴量をデータベースに保存する関数
# def save_features_to_db(conn, features, video_id):
#     cursor = conn.cursor()
#     insert_query = """
#     REPLACE INTO video_features (video_id, num_texts, avg_size, avg_color_r, avg_color_g, avg_color_b, 
#                                   keywords, sentiment, topic_0, topic_1, topic_2, topic_3, topic_4, 
#                                   topic_5, topic_6, topic_7, topic_8, topic_9)
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (
#         video_id, features['num_texts'], features['avg_size'], features['avg_color_r'], 
#         features['avg_color_g'], features['avg_color_b'], features['keywords'], features['sentiment'], 
#         features['topic_0'], features['topic_1'], features['topic_2'], features['topic_3'], features['topic_4'], 
#         features['topic_5'], features['topic_6'], features['topic_7'], features['topic_8'], features['topic_9']
#     ))
#     conn.commit()

# # test_video_folder 内の動画ファイルのパスを取得する関数
# def get_video_test_paths():
#     print("Collecting video paths from test_video_folder...")
#     video_test_paths = [os.path.join(test_video_folder, f) for f in os.listdir(test_video_folder) if f.endswith('.mp4')]
#     print(f"Found {len(video_test_paths)} video files.")
#     return video_test_paths

# # 新しいテスト動画から特徴量を抽出
# def extract_features_from_test_videos():
#     print("Preparing topic modeling for new test videos...")
#     all_texts = []

#     # フォルダ内の動画を取得
#     video_files = [f for f in os.listdir(test_video_folder) if f.endswith('.mp4')]
    
#     # すべての動画からテキストを抽出し、トピックモデリングの準備
#     for video_file in video_files:
#         video_test_path = os.path.join(test_video_folder, video_file)
#         text_data = extract_text_from_video(video_test_path)
#         combined_text = ' '.join([t['text'] for t in text_data])
#         all_texts.append(combined_text)

#     tokenized_texts = [preprocess_text(text) for text in all_texts]
#     dictionary = corpora.Dictionary(tokenized_texts)
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
#     lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

#     print("Extracting features from new test videos...")
#     test_video_features = []

#     conn = connect_to_database()

#     for video_file in video_files:
#         video_test_path = os.path.join(test_video_folder, video_file)
#         frame_count = get_frame_count(video_test_path)  # フレーム数を取得する関数の呼び出し
#         features = extract_features_from_video(video_test_path, frame_count, dictionary, lda_model)
#         test_video_features.append(features)

#         # 動画IDを推測または生成して保存
#         video_id = int(os.path.splitext(video_file)[0])  # ファイル名からIDを取得する仮の方法
#         save_features_to_db(conn, features, video_id)

#     conn.close()
#     print("Feature extraction for test videos completed and saved to database.")

# # 実行
# if __name__ == "__main__":
#     # 最初に動画パスを取得
#     video_test_paths = get_video_test_paths()
    
#     # tiktok_testvideo フォルダ内の動画に対して実行
#     extract_features_from_test_videos(video_test_paths)
