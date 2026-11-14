#!/usr/bin/env python3
"""Relais usbmux entre xtool (WSL) et le service Apple Mobile Device d'iTunes (Windows).

Tant qu'une connexion `Listen` est ouverte, le service Apple ne répond plus aux `ListDevices` suivants, et xtool échoue
en `noDevice`. Le relais répond donc lui-même aux `Listen` (événements Attached/Detached calculés en interrogeant
périodiquement `ListDevices`) ; toutes les autres connexions sont transmises telles quelles.

Usage : usbmux_relay.py <hôte Windows> [port amont=27015] [port local=27016]
"""
import plistlib
import select
import socket
import struct
import sys
import threading

HEADER = struct.Struct("<IIII")  # longueur totale, version, type de message, tag
PLIST_MESSAGE = 8
CLIENT_INFO = {"ClientVersionString": "usbmux_relay", "ProgName": "usbmux_relay", "kLibUSBMuxVersion": 3}


def recv_exact(sock, count):
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("connexion fermée")
        data += chunk
    return data


def read_message(sock):
    """Renvoie (tag, dictionnaire ou None si ce n'est pas un plist, octets bruts du message)."""
    head = recv_exact(sock, HEADER.size)
    length, _, kind, tag = HEADER.unpack(head)
    body = recv_exact(sock, length - HEADER.size)
    return tag, plistlib.loads(body) if kind == PLIST_MESSAGE else None, head + body


def write_message(sock, message, tag):
    body = plistlib.dumps(message)
    sock.sendall(HEADER.pack(HEADER.size + len(body), 1, PLIST_MESSAGE, tag) + body)


def list_devices(upstream):
    with socket.create_connection(upstream, timeout=5) as sock:
        write_message(sock, {"MessageType": "ListDevices", **CLIENT_INFO}, 1)
        return read_message(sock)[1].get("DeviceList", [])


def serve_listen(client, tag, upstream):
    write_message(client, {"MessageType": "Result", "Number": 0}, tag)
    known = {}
    while True:
        try:
            devices = {d["DeviceID"]: d for d in list_devices(upstream)}
        except OSError:
            devices = known  # coupure passagère : on garde l'état connu
        for device_id, device in devices.items():
            if device_id not in known:
                write_message(client, {**device, "MessageType": "Attached"}, 0)
        for device_id in known:
            if device_id not in devices:
                write_message(client, {"MessageType": "Detached", "DeviceID": device_id}, 0)
        known = devices
        if select.select([client], [], [], 1.0)[0] and not client.recv(4096):
            return


def pump(source, destination):
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except OSError:
        pass
    for sock in (source, destination):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def handle(client, upstream):
    with client:
        try:
            tag, message, raw = read_message(client)
            if message and message.get("MessageType") == "Listen":
                serve_listen(client, tag, upstream)
                return
            with socket.create_connection(upstream, timeout=5) as server:
                server.settimeout(None)
                server.sendall(raw)
                other = threading.Thread(target=pump, args=(server, client), daemon=True)
                other.start()
                pump(client, server)
                other.join()
        except (OSError, ConnectionError, plistlib.InvalidFileException):
            pass


def main():
    host = sys.argv[1]
    upstream = (host, int(sys.argv[2]) if len(sys.argv) > 2 else 27015)
    local_port = int(sys.argv[3]) if len(sys.argv) > 3 else 27016
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", local_port))
        listener.listen(32)
        while True:
            client, _ = listener.accept()
            threading.Thread(target=handle, args=(client, upstream), daemon=True).start()


if __name__ == "__main__":
    main()
