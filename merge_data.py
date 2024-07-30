import os
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import configparser
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import signal
import time

# # buzzAI.py から変数をインポート
from buzzAI import numeric_columns, text_columns, date_columns

# 実行時間の上限（秒）
TIME_LIMIT = 1800  # 30分

# タイムアウト関数
def handler(signum, frame):
    print("Execution time exceeded the limit. Terminating the process.")
    raise SystemExit

# タイムアウトを設定
signal.signal(signal.SIGALRM, handler)
signal.alarm(TIME_LIMIT)

# 開始時間
start_time = time.time()

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

# video_id を数値型に変換
video_data['video_id'] = pd.to_numeric(video_data['video_id'], errors='coerce')

# データを結合
merged_data = pd.merge(video_data[['video_id', 'frame_count']], numeric_data, left_on='video_id', right_on='id', how='inner')
merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
merged_data = pd.merge(merged_data, video_features_df, on='video_id', how='inner')

# 日付データをエポック時間に変換
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col])
    merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# 欠損値の補完
numeric_features = merged_data.select_dtypes(include=[np.number])
categorical_features = merged_data.select_dtypes(exclude=[np.number])

imputer_numeric = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

numeric_features_imputed = imputer_numeric.fit_transform(numeric_features)
categorical_features_imputed = imputer_categorical.fit_transform(categorical_features)

# OneHotEncoderを使用して非数値データを数値データに変換
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical_features = encoder.fit_transform(categorical_features_imputed)

# 数値データとエンコードされた非数値データを再結合
X = np.hstack((numeric_features_imputed, encoded_categorical_features))
y = merged_data['動画視聴数']

# 特徴量選択
selector = SelectKBest(f_regression, k='all')
X_selected = selector.fit_transform(X, y)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# モデル選択と学習
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,)),
    'XGBoost': XGBRegressor(),
    # 'LightGBM': LGBMRegressor()
}

param_grids = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'learning_rate_init': [0.001, 0.01]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    # 'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    y_pred = best_models[model_name].predict(X_test)
    print(f"{model_name} - MSE: {mean_squared_error(y_test, y_pred)}, R2: {r2_score(y_test, y_pred)}, MAE: {mean_absolute_error(y_test, y_pred)}")

# グラフの保存先ディレクトリ
result_dir = '/Users/p10475/BuzzCity/result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 学習曲線のプロットと保存
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
# グラフをファイルに保存
plt.savefig(os.path.join(result_dir, 'learning_curves.png'))
# グラフを表示
plt.show()

# テストデータの最初の5つの動画に対して予測
new_video_features = X_test[:5]
actual_view_counts = y_test.iloc[:5].values
predictions = {}

for model_name, model in best_models.items():
    predictions[model_name] = model.predict(new_video_features)

# テストした動画の予測視聴数を出力
prediction_results = pd.DataFrame({
    'Actual View Count': actual_view_counts
})
for model_name, pred in predictions.items():
    prediction_results[f'{model_name} Predicted View Count'] = pred

# CSVファイルに保存
prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# CSVファイルに保存
prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# テストデータに対する予測結果を可視化
plt.figure(figsize=(12, 6))
x = np.arange(len(y_test))

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    plt.plot(x, y_pred, label=f'{model_name} Predictions')

plt.plot(x, y_test.values, 'k--', label='Actual Values', alpha=0.7)
plt.xlabel('Test Sample Index')
plt.ylabel('View Count')
plt.title('Predicted vs Actual View Counts')
plt.legend()
# グラフをファイルに保存
plt.savefig(os.path.join(result_dir, 'predicted_vs_actual.png'))
# グラフを表示
plt.show()

# アンサンブル学習
voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
voting_regressor.fit(X_train, y_train)

# スタッキングアンサンブル学習
estimators = [('rf', RandomForestRegressor(**best_models['RandomForest'].get_params())),
              ('gb', GradientBoostingRegressor(**best_models['GradientBoosting'].get_params()))]
final_estimator = LinearRegression()
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
stacking_regressor.fit(X_train, y_train)

# 学習曲線のプロットと保存
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
    # グラフをファイルに保存
    plt.savefig(os.path.join(result_dir, f'learning_curve_{name}.png'))
    # グラフを表示
    plt.show()

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
    # グラフをファイルに保存
    plt.savefig(os.path.join(result_dir, f'residuals_{model.__class__.__name__}.png'))
    # グラフを表示
    plt.show()

# 各モデルの評価
for name, model in best_models.items():
    evaluate_model(model, X_test, y_test)

# アンサンブルモデルの評価
evaluate_model(voting_regressor, X_test, y_test)

# スタッキングモデルの評価
evaluate_model(stacking_regressor, X_test, y_test)

# データベース接続を閉じる
conn.close()











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
# # from lightgbm import LGBMRegressor
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
# y = merged_data['動画視聴数']

# # 特徴量選択
# selector = SelectKBest(f_regression, k='all')
# X_selected = selector.fit_transform(X, y)

# # データ分割
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# # モデル選択と学習
# models = {
#     'RandomForest': RandomForestRegressor(),
#     'GradientBoosting': GradientBoostingRegressor(),
#     'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,)),
#     'XGBoost': XGBRegressor(),
#     # 'LightGBM': LGBMRegressor()
# }

# param_grids = {
#     'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
#     'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
#     'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'learning_rate_init': [0.001, 0.01]},
#     'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
#     # 'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
# }

# best_models = {}
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
#     best_models[model_name] = grid_search.best_estimator_
#     y_pred = best_models[model_name].predict(X_test)
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
# # グラフをファイルに保存
# plt.savefig(os.path.join(result_dir, 'learning_curves.png'))
# # グラフを表示
# plt.show()

# # テストデータの最初の5つの動画に対して予測
# new_video_features = X_test[:5]
# actual_view_counts = y_test.iloc[:5].values
# predictions = {}

# for model_name, model in best_models.items():
#     predictions[model_name] = model.predict(new_video_features)

# # テストした動画の予測視聴数を出力
# prediction_results = pd.DataFrame({
#     'Actual View Count': actual_view_counts
# })
# for model_name, pred in predictions.items():
#     prediction_results[f'{model_name} Predicted View Count'] = pred

# # CSVファイルに保存
# prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# # CSVファイルに保存
# prediction_results.to_csv(os.path.join(result_dir, 'prediction_results.csv'), index=False)

# # テストデータに対する予測結果を可視化
# plt.figure(figsize=(12, 6))
# x = np.arange(len(y_test))

# for model_name, model in best_models.items():
#     y_pred = model.predict(X_test)
#     plt.plot(x, y_pred, label=f'{model_name} Predictions')

# plt.plot(x, y_test.values, 'k--', label='Actual Values', alpha=0.7)
# plt.xlabel('Test Sample Index')
# plt.ylabel('View Count')
# plt.title('Predicted vs Actual View Counts')
# plt.legend()
# # グラフをファイルに保存
# plt.savefig(os.path.join(result_dir, 'predicted_vs_actual.png'))
# # グラフを表示
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
#     # グラフをファイルに保存
#     plt.savefig(os.path.join(result_dir, f'learning_curve_{name}.png'))
#     # グラフを表示
#     plt.show()

# # モデルの評価
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     # MAPEの計算時にゼロ除算を回避
#     epsilon = np.finfo(np.float64).eps  # 非常に小さい値（ゼロに近い値）
#     mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
    
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
#     # グラフをファイルに保存
#     plt.savefig(os.path.join(result_dir, f'residuals_{model.__class__.__name__}.png'))
#     # グラフを表示
#     plt.show()

# # 各モデルの評価
# for name, model in best_models.items():
#     evaluate_model(model, X_test, y_test)

# # アンサンブルモデルの評価
# evaluate_model(voting_regressor, X_test, y_test)

# # スタッキングモデルの評価
# evaluate_model(stacking_regressor, X_test, y_test)

# # データベース接続を閉じる
# conn.close()