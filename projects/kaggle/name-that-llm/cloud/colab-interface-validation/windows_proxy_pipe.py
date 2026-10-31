"""Relay a single binary stdio stream to the existing Windows loopback proxy."""
import msvcrt
import os
import socket
import sys
import threading

msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
sock = socket.create_connection(("127.0.0.1", 7890), timeout=15)
sock.settimeout(None)

def send():
    try:
        while data := os.read(sys.stdin.fileno(), 65536):
            sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass

threading.Thread(target=send, daemon=True).start()
try:
    while data := sock.recv(65536):
        view = memoryview(data)
        while view:
            view = view[os.write(sys.stdout.fileno(), view):]
finally:
    sock.close()
