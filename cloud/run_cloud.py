from datetime import datetime
import sqlite3
import socket
import json
import queue
import socketserver

score_map = {"0": 20, "1": 40, "2": 40, "3": 100, "4": 50, "5": 80, "6": "80"}

def http_server():
    PORT = 8082
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving HTTP on port {PORT}...")
        httpd.serve_forever()

def socket_server(conn):
    server_ip = '0.0.0.0'
    device_port = 8080
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((server_ip, device_port))
    server_socket.listen()
    print(f"Listening for devices: {server_ip}:{device_port}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Device connected: {client_address}")
        while True:
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                data = json.loads(data)
                emo_id = str(data["message"])
                score = score_map[emo_id]
                current_time = datetime.now()
                formatted_time = current_time.strftime('%Y-%m-%d %H:%M')
                data = (101, 1, score, formatted_time)
                insert_to_table(conn, data)
                

def insert_to_table(conn, data):
    """
    data: (101, 1, 95.5, '2024-01-08 09:00')
    """
    sql = '''INSERT INTO student_performance (student_id, class_id, focus_score, class_time)
         VALUES (?, ?, ?, ?)'''
    cursor = conn.cursor()
    cursor.execute(sql, data)
    conn.commit()
    print("insert data success. data: {}".format(data))

def create_table(conn):
    cursor = conn.cursor()
    table_name = 'student_performance'
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    if cursor.fetchone():
        print("Table exists.")
    else:
        print("Table does not exist, craete table name {}".format(table_name))
        create_table_query = '''
            CREATE TABLE student_performance (
                id INTEGER PRIMARY KEY,
                student_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                focus_score REAL,
                class_time TEXT
            );
        '''
        cursor.execute(create_table_query)
        conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect('./data/db/student_focus.db')
    create_table(conn)
    # data = (101, 1, 95.5, '2024-01-08 09:00')
    # insert_to_table(conn, data)
    socket_server(conn)
