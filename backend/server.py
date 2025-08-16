import socket
import json

HOST = '127.0.0.1'
PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"Server listening on {HOST}:{PORT}")

while True:
    client, address = server.accept()
    print(f"Connection from {address}")
    
    try:
        while True:
            data = client.recv(1024).decode()
            if not data:
                break
            
            # Parse and print received data
            for line in data.strip().split('\n'):
                if line:
                    try:
                        parsed = json.loads(line)
                        print(f"Received: {parsed}")
                    except json.JSONDecodeError:
                        pass
    except ConnectionResetError:
        print("Client disconnected")
    finally:
        client.close()