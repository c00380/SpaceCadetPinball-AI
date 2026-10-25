import os
import random
import socket
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIRS = [os.environ.get("PINBALL_GAME_DIR") or os.path.join(ROOT, "SpaceCadetPinball"), ROOT]
EXE_PATHS = [
    os.path.join("bin", "Release", "SpaceCadetPinball.exe"),
    os.path.join("build", "Release", "SpaceCadetPinball.exe"),
    os.path.join("bin", "SpaceCadetPinball"),
    os.path.join("build", "SpaceCadetPinball"),
]


def find_exe():
    for game_dir in GAME_DIRS:
        for relative in EXE_PATHS:
            path = os.path.join(game_dir, relative)
            if os.path.exists(path):
                return path
    raise FileNotFoundError("SpaceCadetPinball executable not found; clone, patch and build the game "
                            "(see README.md), or set PINBALL_GAME_DIR.")


def free_port():
    for _ in range(50):
        port = random.randint(20000, 58000)
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


class GameClient:

    def __init__(self, headless=True, port=None):
        last_error = None
        for attempt in range(6):
            try:
                self._start(headless, port or free_port())
                return
            except (RuntimeError, OSError, ConnectionError) as e:
                last_error = e
                self.close()
        raise RuntimeError(f"could not start the game: {last_error}")

    def _start(self, headless, port):
        exe = find_exe()
        self.port = port
        self.sock = None
        mode = "-aitrain" if headless else "-aiplay"
        self.proc = subprocess.Popen(
            [exe, mode, f"-aiport={self.port}"],
            cwd=os.path.dirname(exe),
            stdout=subprocess.DEVNULL if headless else None,
            stderr=subprocess.DEVNULL if headless else None,
        )
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError("game exited during startup")
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port))
                break
            except OSError:
                time.sleep(0.1)
        if self.sock is None:
            raise RuntimeError("could not connect to game")
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(30 if headless else None)
        self.f = self.sock.makefile("rw", newline="\n")
        header = self.f.readline().split()
        if not header or header[0] != "H":
            raise RuntimeError("bad handshake")
        self.sock.settimeout(None)
        self.names = header[1:]
        self.index = {n: i for i, n in enumerate(self.names)}

    def send(self, line):
        self.f.write(line + "\n")
        self.f.flush()

    def recv(self):
        resp = self.f.readline()
        if not resp:
            raise ConnectionError("game closed the connection")
        return np.array(resp.split()[1:], dtype=np.float64)

    def _cmd(self, line):
        self.send(line)
        return self.recv()

    def send_step(self, left, right, frames):
        self.send(f"S {int(left)} {int(right)} {int(frames)}")

    def reset(self):
        return self._cmd("R")

    def step(self, left, right, frames):
        return self._cmd(f"S {int(left)} {int(right)} {int(frames)}")

    def close(self):
        try:
            if getattr(self, "sock", None) is not None and getattr(self, "f", None) is not None:
                self.f.write("Q\n")
                self.f.flush()
        except Exception:
            pass
        try:
            if getattr(self, "sock", None) is not None:
                self.sock.close()
        except Exception:
            pass
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
