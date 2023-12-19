import socket
import logging
from threading import Thread
import json

clients = {}


def client_handler(client_socket, client_ip, logger):
    try:
        while True:
            full_data = client_socket.recv(1024)
            if not full_data:
                break
            # We assume that after the first message, all following data does not contain the identifier
            print("DEBUG: client_handler - Incoming data: {}".format(full_data))

            # Echo the data back to all registered sockets from the same IP address
            logger.info(
                "Middleware_Server - streaming data back: %s", full_data)
            for socket in clients[client_ip]:
                try:
                    # Streaming the data back to the application
                    print(
                        f"INFO: Sending to {client_ip} with socket: {socket}\n")
                    socket.sendall(full_data)
                except Exception as e:
                    logger.error(
                        f"Error sending to one of the clients from {client_ip}: {e}")

    except ConnectionAbortedError:
        logger.error("Connection was aborted.")
    finally:
        client_socket.close()
        clients[client_ip].remove(client_socket)
        if not clients[client_ip]:
            del clients[client_ip]
        logger.info(f"Closed connection with {client_ip}")


def start_server(host, port):

    # Setting up the logging
    log_format = "%(asctime)s - %(name)s - %(message)s"
    logging.basicConfig(filename="G:\\Dissertation_Project\\Logs\\performance_logs.log",
                        level=logging.INFO, format=log_format)
    logger = logging.getLogger("Dissertation_Project")

    # Server setup
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    print(f"Starting server on {host}:{port}")

    while True:
        client_socket, addr = server.accept()
        client_ip = addr[0]

        if client_ip not in clients:
            clients[client_ip] = []

        clients[client_ip].append(client_socket)
        logger.info(f"Connection from {addr} registered.")

        print(f"INFO: start_server - Accepted Connection from {addr}")

        Thread(target=client_handler, args=(
            client_socket, client_ip, logger)).start()


if __name__ == "__main__":
    start_server('localhost', 9999)
