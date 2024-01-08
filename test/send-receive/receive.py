import socket
import json

# 设置服务器的IP地址和端口
server_ip = '0.0.0.0'  # 0.0.0.0 表示绑定到所有可用的接口
server_port = 8080

# 创建socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定socket到指定的IP地址和端口
server_socket.bind((server_ip, server_port))

# 开始监听连接
server_socket.listen()

print(f"服务器启动，监听地址：{server_ip}:{server_port}")

# 接受一个连接
client_socket, client_address = server_socket.accept()
print(f"连接自：{client_address}")

# 接收数据
data = client_socket.recv(1024).decode('utf-8')
print(f"接收到的数据：{data}")

# 关闭连接
client_socket.close()
server_socket.close()
