import sqlite3
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

# 查询特定时间段内数据的SQL
def select_time_range(conn):
    cursor = conn.cursor()
    start_time = '2024-01-08 09:00'
    end_time = '2024-01-09 12:00'
    sql = '''
        SELECT focus_score FROM student_performance
        WHERE class_time BETWEEN ? AND ?;
    '''
    cursor.execute(sql, (start_time, end_time))
    results = cursor.fetchall()
    avg_focus_score = 0
    for row in results:
        avg_focus_score += row[0]
    avg_focus_score /= len(results)
    return avg_focus_score

def generate_datas(conn):
    avg_focus_score = select_time_range(conn)
    datas = {'专注':avg_focus_score, '不专注': 100 - avg_focus_score}
    return datas


@app.route('/')
def index():
    conn = sqlite3.connect('../data/db/student_focus.db')
    datas = generate_datas(conn)
    return render_template('index.html', datas=datas)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)


