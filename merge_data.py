import os
import cv2
import mysql.connector as mydb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import configparser
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
import datetime
import random

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

# データベースから動画情報を読み込み (id 列も読み込む)
cursor.execute("SELECT id, video_id, frame_count FROM videos")
video_data = pd.DataFrame(cursor.fetchall(), columns=['id', 'video_id', 'frame_count'])

# データベースから数値データを読み込み
cursor.execute("SELECT * FROM numeric_data")
numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースからテキストデータを読み込み
cursor.execute("SELECT * FROM text_data")
text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# データベースから日付データを読み込み
cursor.execute("SELECT * FROM date_data")
date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# video_featuresテーブルから特徴量を読み込み
cursor.execute("SELECT * FROM video_features")
video_features_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# video_id を数値型に変換
video_data['video_id'] = pd.to_numeric(video_data['video_id'], errors='coerce')

# データ結合 (動画データがあるものに限定)
merged_data = pd.merge(video_data[['video_id', 'frame_count', 'id']], numeric_data, on='id', how='inner')
merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
merged_data = pd.merge(merged_data, video_features_data, on='video_id', how='inner')

# 日付データをエポック時間に変換
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col])
    merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# 特徴量とターゲットの分割
X = merged_data.drop(['video_id', '動画視聴数'], axis=1)  # video_idと動画視聴数は学習には不要
y = merged_data['動画視聴数']  # ターゲット変数

# ターゲット変数 y に 0 が含まれている場合、削除
X = X[y != 0]
y = y[y != 0]

# 文字列のカラムを抽出
categorical_columns = X.select_dtypes(include=['object']).columns

# 数値のカラムを抽出
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# 列変換器を定義
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # NaN 値を補完
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # NaN 値を補完
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# モデル選択と学習
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'SVM': SVR(),
    'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,))
}

param_grids = {
    'RandomForest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [10, 20, 30]},
    'GradientBoosting': {'model__learning_rate': [0.01, 0.1, 0.2], 'model__n_estimators': [100, 200, 300]},
    'SVM': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
    'NeuralNetwork': {'model__hidden_layer_sizes': [(50,), (100,), (100, 50)], 'model__alpha': [0.0001, 0.001]}
}

best_models = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('selector', SelectKBest(f_regression, k='all')),
                               ('model', model)])
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_models[name] = grid_search.best_estimator_

# アンサンブル学習
voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
voting_regressor.fit(X, y)

# ランダムに動画を選択し、その視聴数を予測
random_video = merged_data.sample(n=1, random_state=42)
X_test = random_video.drop(['video_id', '動画視聴数'], axis=1)
y_test = random_video['動画視聴数']

# 視聴数を予測
y_pred = voting_regressor.predict(X_test)

# 予測と実際の視聴数の差を計算
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Predicted: {y_pred[0]}, Actual: {y_test.values[0]}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# 残差プロット
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'Residuals vs Predicted Values for Voting Regressor')
plt.show()

# データベース接続を閉じる
conn.close()



# import os
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
# from sklearn.impute import SimpleImputer
# import configparser
# import matplotlib.pyplot as plt
# import seaborn as sns
# from category_encoders import TargetEncoder
# import datetime

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

# # データベースから動画情報を読み込み (id 列も読み込む)
# cursor.execute("SELECT id, video_id, frame_count FROM videos")  # id 列も読み込む
# video_data = pd.DataFrame(cursor.fetchall(), columns=['id', 'video_id', 'frame_count'])

# # データベースから数値データを読み込み
# cursor.execute("SELECT * FROM numeric_data")
# numeric_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # データベースからテキストデータを読み込み
# cursor.execute("SELECT * FROM text_data")
# text_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # データベースから日付データを読み込み
# cursor.execute("SELECT * FROM date_data")
# date_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # video_featuresテーブルから特徴量を読み込み
# cursor.execute("SELECT * FROM video_features")
# video_features_data = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

# # video_id を数値型に変換
# video_data['video_id'] = pd.to_numeric(video_data['video_id'], errors='coerce')

# # データ結合
# merged_data = pd.merge(video_data[['video_id', 'frame_count', 'id']], numeric_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, text_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, date_data, on='id', how='inner')
# merged_data = pd.merge(merged_data, video_features_data, on='video_id', how='inner')

# # 日付データをエポック時間に変換
# for col in date_columns:
#     merged_data[col] = pd.to_datetime(merged_data[col])
#     merged_data[col] = merged_data[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.nan)

# # 特徴量とターゲットの分割
# X = merged_data.drop(['video_id', '動画視聴数'], axis=1)  # video_idと動画視聴数は学習には不要
# y = merged_data['動画視聴数']  # ターゲット変数

# # ターゲット変数 y に 0 が含まれている場合、削除
# X = X[y != 0]
# y = y[y != 0]

# # 文字列のカラムを抽出
# categorical_columns = X.select_dtypes(include=['object']).columns

# # 数値のカラムを抽出
# numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# # 列変換器を定義
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),  # NaN 値を補完
#     ('scaler', StandardScaler()),
#     ('power', PowerTransformer(method='yeo-johnson'))
# ])
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),  # NaN 値を補完
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_columns),
#         ('cat', categorical_transformer, categorical_columns)
#     ])

# # モデル選択と学習
# models = {
#     'RandomForest': RandomForestRegressor(),
#     'GradientBoosting': GradientBoostingRegressor(),
#     'SVM': SVR(),
#     'NeuralNetwork': MLPRegressor(max_iter=1000, learning_rate_init=0.001, hidden_layer_sizes=(100,))
# }

# param_grids = {
#   'RandomForest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [10, 20, 30]},
#   'GradientBoosting': {'model__learning_rate': [0.01, 0.1, 0.2], 'model__n_estimators': [100, 200, 300]},
#   'SVM': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
#   'NeuralNetwork': {'model__hidden_layer_sizes': [(50,), (100,), (100, 50)], 'model__alpha': [0.0001, 0.001]}
# }

# best_models = {}
# for name, model in models.items():
#     # パイプラインで前処理とモデルを結合
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                 ('selector', SelectKBest(f_regression, k='all')),  # 特徴量選択
#                                 ('model', model)])
#     grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error')
#     grid_search.fit(X, y)  # 全データをパイプラインに適用
#     best_models[name] = grid_search.best_estimator_

# # アンサンブル学習
# voting_regressor = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
# voting_regressor.fit(X, y)  # 全データで学習

# # スタッキングアンサンブル学習
# estimators = [('rf', RandomForestRegressor(**best_models['RandomForest'].get_params())),
#               ('gb', GradientBoostingRegressor(**best_models['GradientBoosting'].get_params()))]
# final_estimator = LinearRegression()
# stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
# stacking_regressor.fit(X, y)

# # 学習曲線のプロット
# models_for_learning_curve = {
#     'VotingRegressor': voting_regressor,
#     'StackingRegressor': stacking_regressor
# }
# for name, model in models_for_learning_curve.items():
#     train_sizes, train_scores, test_scores = learning_curve(
#         model, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
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
#     plt.show()

# # モデルの評価
# def evaluate_model(model, X, y):
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)
#     mae = mean_absolute_error(y, y_pred)
#     # MAPEの計算時にゼロ除算を回避
#     epsilon = np.finfo(np.float64).eps  
#     mape = np.mean(np.abs((y - y_pred) / (y + epsilon))) * 100 

#     print(f'Model: {model.__class__.__name__}')
#     print(f'Mean Squared Error: {mse}')
#     print(f'R^2 Score: {r2}')
#     print(f'Mean Absolute Error: {mae}')
#     print(f'Mean Absolute Percentage Error: {mape}')

#     # 残差プロット
#     residuals = y - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_pred, residuals)
#     plt.axhline(0, color='red', linestyle='--')
#     plt.xlabel('Predicted Values')
#     plt.ylabel('Residuals')
#     plt.title(f'Residuals vs Predicted Values for {model.__class__.__name__}')
#     plt.show()

# # 各モデルの評価
# for name, model in best_models.items():
#     evaluate_model(model, X, y)

# # アンサンブルモデルの評価
# evaluate_model(voting_regressor, X, y)

# # スタッキングモデルの評価
# evaluate_model(stacking_regressor, X, y)

# # データベース接続を閉じる
# conn.close()
