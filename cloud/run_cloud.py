import asyncio
import websockets
import socket
import json
import queue
import http.server
import socketserver

# 创建一个队列来存储接收到的数据
data_queue = queue.Queue()

async def echo(websocket, path):
    while True:
        # 持续发送队列中的所有数据，直到队列为空
        while not data_queue.empty():
            data = data_queue.get()
            await websocket.send(json.dumps(data))
        # 发送完毕后休息一秒
        await asyncio.sleep(1)

def http_server():
    PORT = 8082
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving HTTP on port {PORT}...")
        httpd.serve_forever()

def socket_server():
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
                print(f"Received from device: {data}")
                # 将接收到的数据放入队列中
                data_queue.put(json.loads(data))

# 启动 WebSocket 服务器
start_server = websockets.serve(echo, "0.0.0.0", 8081)

async def websocket_server():
    async with websockets.serve(echo, "0.0.0.0", 8081):
        await asyncio.Future()  # 这将使得 WebSocket 服务器持续运行

# 使用 asyncio 运行两个服务器
async def main():
    task1 = asyncio.create_task(websocket_server())
    task2 = asyncio.to_thread(socket_server)
    task3 = asyncio.to_thread(http_server)
    await asyncio.gather(task1, task2, task3)

asyncio.run(main())
