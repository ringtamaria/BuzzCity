import mysql.connector as mydb

# # コネクションの作成
# conn = mydb.connect(
#     host='localhost',
#     port='3306',
#     user='rintamaria',
#     password='buzzai',
#     database='mydb'
# )

# # コネクションが切れた時に再接続してくれるよう設定します。
# conn.ping(reconnect=True)

# # DB操作用にカーソルを作成
# cur = conn.cursor()

 
# # コミットしてトランザクション実行
# conn.commit()

# # カーソルとコネクションを閉じる
# cur.close()
# conn.close()

try:
    conn = mydb.connect(
        host="localhost",
        port='3306',
        user="rintamaria",
        password="buzzai",
        database="mydb"
    )
    cursor = conn.cursor()

    # テーブルの作成（必要に応じて）
    cursor.execute("CREATE TABLE IF NOT EXISTS test (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255))")
    
    # データの挿入
    cursor.execute("INSERT INTO test (name) VALUES ('Alice')")
    cursor.execute("INSERT INTO test (name) VALUES ('Bob')")
    conn.commit()

    # データの取得
    cursor.execute("SELECT * FROM test")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    print("Operation successful")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("Connection closed")