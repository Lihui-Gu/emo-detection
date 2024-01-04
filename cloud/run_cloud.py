import socket
import json

server_ip = '0.0.0.0'
server_port = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind((server_ip, server_port))

server_socket.listen()

print(f"Listening: {server_ip}:{server_port}")

client_socket, client_address = server_socket.accept()
print(f"connect: {client_address}")

data = client_socket.recv(1024).decode('utf-8')
print(f"receive: {data}")

client_socket.close()
server_socket.close()
