import mysql.connector as mydb
import pandas as pd
import numpy as np

# CSVファイルの読み込み
file_path = '/Users/p10475/BuzzCity/to buy案件数値レポート - to buy案件数値.csv'
data = pd.read_csv(file_path, skiprows=4, nrows=188)

# データの確認
print(data.head().to_markdown(index=False, numalign='left', stralign='left'))

# データにIDを追加する
data['id'] = range(1, len(data) + 1)

# カラム名の変更 (変更前の名前 -> 変更後の名前)
data = data.rename(columns={
    '掲載単価(N)': '掲載単価N',
    '75%(動画が再生された長さが)': '再生長さ75パーセント',
    '50%(動画が再生された長さが)': '再生長さ50パーセント',
    '25%(動画が再生された長さが)': '再生長さ25パーセント',
    'オーガニック再生数（実績ー広告配信mp）': 'オーガニック再生数'
})

# カラム名の確認
print("Modified Column Names:")
print(data.columns.tolist())

# 不要な列 `Unnamed: 0` を削除
data = data.drop(columns=['Unnamed: 0'])

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
                   'シェア',
                   'コメント率',
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

# 重複している列を削除
numeric_data = data[numeric_columns].copy()
numeric_data = numeric_data.loc[:,~numeric_data.columns.duplicated()]
text_data = data[text_columns].copy()
date_data = data[date_columns].copy()

# データの型変換（必要な場合）
date_data[date_columns] = date_data[date_columns].apply(pd.to_datetime, errors='coerce')

# 欠損値と#DIV/0!の処理
# 数値データの欠損値と#DIV/0!をNaNに変換
numeric_data.replace(to_replace='#DIV/0!', value=np.nan, inplace=True)
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# テキストデータの欠損値と#DIV/0!をNoneに変換
text_data.replace(to_replace='#DIV/0!', value=None, inplace=True)
text_data = text_data.where(pd.notnull(text_data), None)

# 日付データの欠損値をNaTに変換
date_data.replace(to_replace='#DIV/0!', value=np.nan, inplace=True)
date_data = date_data.where(pd.notnull(date_data), None)

# 欠損値と#DIV/0!の処理 (NaNをNULLに変換)
numeric_data = numeric_data.fillna(np.nan).replace([np.nan], [None])
text_data = text_data.fillna(np.nan).replace([np.nan], [None])
date_data = date_data.fillna(np.nan).replace([np.nan], [None])


# データベース接続
try:
    conn = mydb.connect(
        host="localhost",
        port='3306',
        user="rintamaria",
        password="buzzai",
        database="mydb"
    )

    cursor = conn.cursor()

    # 既存のテーブルを削除
    cursor.execute("DROP TABLE IF EXISTS numeric_data")
    cursor.execute("DROP TABLE IF EXISTS text_data")
    cursor.execute("DROP TABLE IF EXISTS date_data")

    # テーブル作成（もし存在しなければ）
    create_numeric_table_query = f"""
    CREATE TABLE IF NOT EXISTS numeric_data (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        {', '.join([f'`{col}` FLOAT' for col in numeric_data.columns])}
    );
    """
    create_text_table_query = f"""
    CREATE TABLE IF NOT EXISTS text_data (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        {', '.join([f'`{col}` VARCHAR(255)' for col in text_columns])}
    );
    """
    create_date_table_query = f"""
    CREATE TABLE IF NOT EXISTS date_data (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        {', '.join([f'`{col}` DATE' for col in date_columns])}
    );
    """

    cursor.execute(create_numeric_table_query)
    cursor.execute(create_text_table_query)
    cursor.execute(create_date_table_query)

    # データ挿入
    insert_numeric_query = f"""
    INSERT INTO numeric_data ({', '.join([f'`{col}`' for col in numeric_data.columns])}) 
    VALUES ({', '.join(['%s'] * len(numeric_data.columns))})
    """
    for _, row in numeric_data.iterrows():
        cursor.execute(insert_numeric_query, tuple(row))

    insert_text_query = f"""
    INSERT INTO text_data ({', '.join([f'`{col}`' for col in text_data.columns])}) 
    VALUES ({', '.join(['%s'] * len(text_data.columns))})
    """
    for _, row in text_data.iterrows():
        cursor.execute(insert_text_query, tuple(row))

    insert_date_query = f"""
    INSERT INTO date_data ({', '.join([f'`{col}`' for col in date_data.columns])}) 
    VALUES ({', '.join(['%s'] * len(date_data.columns))})
    """
    for _, row in date_data.iterrows():
        cursor.execute(insert_date_query, tuple(row))

    # コミットしてトランザクション実行
    conn.commit()

    print("Data import successful")

except mydb.Error as err:  # エラーハンドリング (インデント修正)
    print(f"Something went wrong: {err}")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("Connection closed")
