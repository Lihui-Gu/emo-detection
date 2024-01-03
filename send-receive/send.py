import socket
import json

# 服务器的IP地址和端口
server_ip = '106.13.201.126'  # 替换为服务器的实际IP地址
server_port = 12345

# 创建JSON数据
data = {"message": "Hello, Server!"}
json_data = json.dumps(data)

# 创建socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接到服务器
client_socket.connect((server_ip, server_port))

# 发送数据
client_socket.sendall(json_data.encode('utf-8'))

# 关闭socket
client_socket.close()
