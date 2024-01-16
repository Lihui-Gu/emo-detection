import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify

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

@app.route('/get_options', methods=['POST'])
def get_options():
    conn = sqlite3.connect('../data/db/student_focus.db')
    cursor = conn.cursor()
    sql_class_id = '''
        SELECT DISTINCT class_id FROM student_performance;
    '''
    cursor.execute(sql_class_id)
    class_id_results = cursor.fetchall()
    class_id_results = [x[0] for x in class_id_results]
    conn.commit()
    sql_class_time = '''
        SELECT DISTINCT class_time FROM student_performance;
    '''
    cursor.execute(sql_class_time)
    class_time_results = cursor.fetchall()
    class_time_results = [x[0] for x in class_time_results]
    conn.commit()
    cursor.close()
    conn.close()
    options_data = {"class_id_results": class_id_results, "class_time_results": class_time_results}
    print(options_data)
    return options_data

@app.route('/search', methods=['POST'])
def search():
    class_id = request.form['class_id']
    class_time = request.form['class_time']

    # 连接数据库
    conn = sqlite3.connect('../data/db/student_focus.db')
    try:
        cursor = conn.cursor()
        sql = '''
            SELECT focus_score FROM student_performance
            WHERE class_id=? AND class_time=?;
        '''
        cursor.execute(sql, (class_id, class_time))
        focus_score_results = cursor.fetchall()
        focus_score_results = [x[0] for x in focus_score_results]
        conn.commit()
        avg_focus_score = 0
        for score in focus_score_results:
            avg_focus_score += score
        avg_focus_score /= len(focus_score_results)
        datas = [{"value": avg_focus_score, "name": "专注"}, {"value": 100 - avg_focus_score, "name": "不专注"}]
        print(avg_focus_score)
        return datas
    except Exception as e:
        print("not success")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)


