import sqlite3


if __name__ == "__main__":
    conn = sqlite3.connect('example.db')
    # 创建一个游标对象
    cursor = conn.cursor()
    # 创建表
    cursor.execute('''CREATE TABLE IF NOT EXISTS stocks
                  (date text, trans text, symbol text, qty real, price real)''')
    # 插入一行数据
    cursor.execute("INSERT INTO stocks VALUES ('2024-01-08','BUY','RHAT',100,35.14)")
    # 提交当前事务
    conn.commit()
    # 关闭连接
    conn.close()
