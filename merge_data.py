# import os
# import mysql.connector as mydb
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.impute import SimpleImputer
# import configparser
# import matplotlib.pyplot as plt
# from category_encoders import TargetEncoder
# from xgboost import XGBRegressor
# import signal
# import time

# # # buzzAI.py から変数をインポート
# from buzzAI import numeric_columns, text_columns, date_columns

# # 実行時間の上限（秒）
# TIME_LIMIT = 1800  # 30分

# # タイムアウト関数
# def handler(signum, frame):
#     print("Execution time exceeded the limit. Terminating the process.")
#     raise SystemExit

# # タイムアウトを設定
# signal.signal(signal.SIGALRM, handler)
# signal.alarm(TIME_LIMIT)

# # 開始時間
# start_time = time.time()

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

# # データベースから動画特徴量データを読み込み
# cursor.execute("SELECT * FROM video_features")
# video_features_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# def handle_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df[column] = np.where(df[column] > upper_bound, upper_bound,
#                           np.where(df[column] < lower_bound, lower_bound, df[column]))

# # データ前処理
# for col in numeric_columns:
#     handle_outliers(numeric_data, col)

# # テキストデータのエンコーディング
# encoder = TargetEncoder()
# for col in text_columns:
#     if col in text_data.columns:
#         text_data[f'encoded_{col}'] = encoder.fit_transform(text_data[col], numeric_data['動画視聴数'])

# # テキストカラムの削除（エンコード後は元のテキストカラムは不要）
# text_data = text_data.drop(columns=text_columns, errors='ignore')

# # video_id を数値型に変換
# video_data['video_id'] = pd.to_numeric(video_data['video_id'], errors='coerce')

# # データを結合
# merged_data = pd.merge(video_data[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
# merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')

# # 日付データをエポック時間に変換
# for col in date_columns:
#     merged_data[col] = pd.to_datetime(merged_data[col])
#     merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# # 欠損値の補完
# numeric_features = merged_data.select_dtypes(include=[np.number])
# categorical_features = merged_data.select_dtypes(exclude=[np.number])

# imputer_numeric = SimpleImputer(strategy='mean')
# imputer_categorical = SimpleImputer(strategy='most_frequent')

# numeric_features_imputed = imputer_numeric.fit_transform(numeric_features)
# categorical_features_imputed = imputer_categorical.fit_transform(categorical_features)

# # OneHotEncoderを使用して非数値データを数値データに変換
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# encoded_categorical_features = encoder.fit_transform(categorical_features_imputed)

# # 数値データとエンコードされた非数値データを再結合
# X = np.hstack((numeric_features_imputed, encoded_categorical_features))

# # ターゲット変数のスケーリング
# y = merged_data['動画視聴数'].values.reshape(-1, 1)
# target_scaler = StandardScaler()
# y_scaled = target_scaler.fit_transform(y).flatten()

# # 特徴量選択
# selector = SelectKBest(f_regression, k='all')
# X_selected = selector.fit_transform(X, y_scaled)

# # データ分割
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y_scaled, test_size=0.2, random_state=42)

# # モデル選択と学習
# models = {
#     'RandomForest': RandomForestRegressor(),
#     'GradientBoosting': GradientBoostingRegressor(),
#     'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,)),
#     'XGBoost': XGBRegressor()
# }

# param_grids = {
#     'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
#     'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
#     'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'learning_rate_init': [0.001, 0.01]},
#     'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
# }

# best_models = {}
# predictions = {}
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
#     best_models[model_name] = grid_search.best_estimator_
#     y_pred = best_models[model_name].predict(X_test)
    
#     # 予測値の逆スケーリングとマイナス値の処理
#     y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
#     y_pred = np.maximum(y_pred, 0)
    
#     predictions[model_name] = y_pred
    
#     print(f"{model_name} - MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}, MAE: {mean_absolute_error(y_test, y_pred)}")

# # グラフの保存先ディレクトリ
# result_dir = '/Users/p10475/BuzzCity/result'
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

# # 学習曲線のプロットと保存
# plt.figure(figsize=(10, 5))
# for model_name, model in best_models.items():
#     train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
#     plt.plot(train_sizes, -train_scores.mean(axis=1), label=f'{model_name} Train')
#     plt.plot(train_sizes, -test_scores.mean(axis=1), label=f'{model_name} Test')

# plt.xlabel('Training examples')
# plt.ylabel('Mean Squared Error')
# plt.title('Learning Curves for Different Models')
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(result_dir, 'learning_curves.png'))
# plt.show()

# # テストデータの最初の5つの動画に対して予測
# new_video_features = X_test[:5]
# actual_view_counts = y_test[:5]

# # テストした動画の予測視聴数を出力
# prediction_results = pd.DataFrame({
#     'Actual View Count': target_scaler.inverse_transform(actual_view_counts.reshape(-1, 1)).flatten()
# })
# for model_name, pred in predictions.items():
#     prediction_results[f'{model_name} Predicted View Count'] = pred[:5]

# # CSVファイルに保存
# prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# # テストデータに対する予測結果を可視化
# plt.figure(figsize=(12, 6))
# x = np.arange(len(y_test))

# for model_name, y_pred in predictions.items():
#     plt.plot(x, y_pred, label=f'{model_name} Predictions')

# plt.plot(x, target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'k--', label='Actual Values', alpha=0.7)
# plt.xlabel('Test Sample Index')
# plt.ylabel('View Count')
# plt.title('Predicted vs Actual View Counts')
# plt.legend()
# plt.savefig(os.path.join(result_dir, 'predicted_vs_actual.png'))
# plt.show()

# # アンサンブル学習
# voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
# voting_regressor.fit(X_train, y_train)

# # スタッキングアンサンブル学習
# estimators = [('rf', RandomForestRegressor(**best_models['RandomForest'].get_params())),
#               ('gb', GradientBoostingRegressor(**best_models['GradientBoosting'].get_params()))]
# final_estimator = LinearRegression()
# stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
# stacking_regressor.fit(X_train, y_train)

# # 学習曲線のプロットと保存
# models_for_learning_curve = {
#     'VotingRegressor': voting_regressor,
#     'StackingRegressor': stacking_regressor
# }
# for name, model in models_for_learning_curve.items():
#     train_sizes, train_scores, test_scores = learning_curve(
#         model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
#     )

#     train_scores_mean = -train_scores.mean(axis=1)
#     test_scores_mean = -test_scores.mean(axis=1)

#     plt.figure()
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#     plt.title(f'Learning Curve for {name}')
#     plt.xlabel('Training Size')
#     plt.ylabel('MSE')
#     plt.legend(loc="best")
#     plt.savefig(os.path.join(result_dir, f'learning_curve_{name}.png'))
#     plt.show()

# def evaluate_model(model, X_test, y_test, model_name):
#     y_pred = model.predict(X_test)
#     y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
#     y_pred = np.maximum(y_pred, 0)
#     y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
#     mse = mean_squared_error(y_test_original, y_pred)
#     r2 = r2_score(y_test_original, y_pred)
#     mae = mean_absolute_error(y_test_original, y_pred)
#     mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100 if not (y_test_original == 0).any() else np.inf
    
#     print(f'Model: {model_name}')
#     print(f'Mean Squared Error: {mse}')
#     print(f'R^2 Score: {r2}')
#     print(f'Mean Absolute Error: {mae}')
#     print(f'Mean Absolute Percentage Error: {mape}')
    
#     # 残差プロット
#     residuals = y_test_original - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_pred, residuals)
#     plt.axhline(0, color='red', linestyle='--')
#     plt.xlabel('Predicted Values')
#     plt.ylabel('Residuals')
#     plt.title(f'Residuals vs Predicted Values for {model_name}')
#     plt.savefig(os.path.join(result_dir, f'residuals_{model_name}.png'))
#     plt.show()

# # 各モデルの評価
# for name, model in best_models.items():
#     evaluate_model(model, X_test, y_test, name)

# # アンサンブルモデルの評価
# evaluate_model(voting_regressor, X_test, y_test, 'VotingRegressor')

# # スタッキングモデルの評価
# evaluate_model(stacking_regressor, X_test, y_test, 'StackingRegressor')

# # データベース接続を閉じる
# conn.close()


import os
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression, Lasso  # LinearRegressionをインポート
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

# データベースから動画情報を読み込み
print_progress("Loading video data from the database...")
cursor.execute("SELECT video_id, video_path, frame_count FROM videos")
video_data = pd.DataFrame(cursor.fetchall(), columns=['video_id', 'video_path', 'frame_count'])

# データベースから数値データを読み込み
print_progress("Loading numeric data from the database...")
cursor.execute("SELECT * FROM numeric_data")
numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースからテキストデータを読み込み
print_progress("Loading text data from the database...")
cursor.execute("SELECT * FROM text_data")
text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースから日付データを読み込み
print_progress("Loading date data from the database...")
cursor.execute("SELECT * FROM date_data")
date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースから動画特徴量データを読み込み
print_progress("Loading video features data from the database...")
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
print_progress("Handling outliers...")
for col in numeric_columns:
    handle_outliers(numeric_data, col)

# テキストデータのエンコーディング
print_progress("Encoding text data...")
encoder = TargetEncoder()
for col in text_columns:
    if col in text_data.columns:
        text_data[f'encoded_{col}'] = encoder.fit_transform(text_data[col], numeric_data['動画視聴数'])

# テキストカラムの削除（エンコード後は元のテキストカラムは不要）
text_data = text_data.drop(columns=text_columns, errors='ignore')

# video_id を数値型に変換
print_progress("Converting video_id to numeric...")
video_data['video_id'] = pd.to_numeric(video_data['video_id'], errors='coerce')

# データを結合
print_progress("Merging data...")
merged_data = pd.merge(video_data[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')

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

# データ分割
print_progress("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_scaled, test_size=0.2, random_state=42)

# モデル選択と学習
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

param_grids = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

best_models = {}
predictions = {}
for model_name, model in models.items():
    print_progress(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    y_pred = best_models[model_name].predict(X_test)
    
    # 予測値の逆スケーリングとマイナス値の処理
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)
    
    predictions[model_name] = y_pred
    
    print(f"{model_name} - MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}, MAE: {mean_absolute_error(y_test, y_pred)}")

# グラフの保存先ディレクトリ
result_dir = '/Users/p10475/BuzzCity/result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 学習曲線のプロットと保存
print_progress("Plotting learning curves...")
plt.figure(figsize=(10, 5))
for model_name, model in best_models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    plt.plot(train_sizes, -train_scores.mean(axis=1), label=f'{model_name} Train')
    plt.plot(train_sizes, -test_scores.mean(axis=1), label=f'{model_name} Test')

plt.xlabel('Training examples')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves for Different Models')
plt.legend()
plt.grid()
plt.savefig(os.path.join(result_dir, 'learning_curves.png'))
plt.show()

# テストデータの最初の5つの動画に対して予測
print_progress("Predicting view counts for test data...")
new_video_features = X_test[:5]
actual_view_counts = y_test[:5]

# テストした動画の予測視聴数を出力
prediction_results = pd.DataFrame({
    'Actual View Count': target_scaler.inverse_transform(actual_view_counts.reshape(-1, 1)).flatten()
})
for model_name, pred in predictions.items():
    prediction_results[f'{model_name} Predicted View Count'] = pred[:5]

# CSVファイルに保存
prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# テストデータに対する予測結果を可視化
print_progress("Visualizing predictions vs actual view counts...")
plt.figure(figsize=(12, 6))
x = np.arange(len(y_test))

for model_name, y_pred in predictions.items():
    plt.plot(x, y_pred, label=f'{model_name} Predictions')

plt.plot(x, target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 'k--', label='Actual Values', alpha=0.7)
plt.xlabel('Test Sample Index')
plt.ylabel('View Count')
plt.title('Predicted vs Actual View Counts')
plt.legend()
plt.savefig(os.path.join(result_dir, 'predicted_vs_actual.png'))
plt.show()

# アンサンブル学習
print_progress("Training Voting Regressor...")
voting_regressor = VotingRegressor(estimators=[('RandomForest', best_models['RandomForest']),
                                               ('GradientBoosting', best_models['GradientBoosting']),
                                               ('XGBoost', best_models['XGBoost'])])
voting_regressor.fit(X_train, y_train)

# スタッキングアンサンブル学習
print_progress("Training Stacking Regressor...")
estimators = [('rf', RandomForestRegressor(**best_models['RandomForest'].get_params())),
              ('gb', GradientBoostingRegressor(**best_models['GradientBoosting'].get_params()))]
final_estimator = LinearRegression()
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
stacking_regressor.fit(X_train, y_train)

# 学習曲線のプロットと保存
print_progress("Plotting learning curves for ensemble models...")
models_for_learning_curve = {
    'VotingRegressor': voting_regressor,
    'StackingRegressor': stacking_regressor
}
for name, model in models_for_learning_curve.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
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
    plt.savefig(os.path.join(result_dir, f'learning_curve_{name}.png'))
    plt.show()

# モデルの評価
def evaluate_model(model, X_test, y_test, model_name):
    print_progress(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100 if not (y_test_original == 0).any() else np.inf
    
    print(f'Model: {model_name}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')
    
    # 残差プロット
    print_progress(f"Plotting residuals for {model_name}...")
    residuals = y_test_original - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for {model_name}')
    plt.savefig(os.path.join(result_dir, f'residuals_{model_name}.png'))
    plt.show()

# 各モデルの評価
for name, model in best_models.items():
    evaluate_model(model, X_test, y_test, name)

# アンサンブルモデルの評価
evaluate_model(voting_regressor, X_test, y_test, 'VotingRegressor')
evaluate_model(stacking_regressor, X_test, y_test, 'StackingRegressor')

# データベース接続を閉じる
print_progress("Closing database connection...")
conn.close()
print_progress("Process completed.")

