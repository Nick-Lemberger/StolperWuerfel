#!/usr/bin/env python3
"""UDP-Ansteuerung des LED-Würfels (6 Panels à 64x64 RGB).

Paketformat (unverändert): [panel+1, zeile] + 64 Pixel * (B, G, R)
"""
import socket
import numpy as np


class Cube:
    num_rows = 64
    num_cols = 64
    panel_num = 6
    header_size = 2
    row_bytes = header_size + num_cols * 3          # 194 Byte pro Paket

    def __init__(self, UDP_IP, UDP_PORT):
        self.UDP_IP = UDP_IP
        self.UDP_PORT = UDP_PORT
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Größerer Sendepuffer, damit 384 Pakete pro Frame nicht blockieren
        try:
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError:
            pass
        # connect() bei UDP: Zieladresse wird einmal aufgelöst, danach send() statt sendto()
        self.s.connect((UDP_IP, UDP_PORT))

        # Vorbereiteter Paketpuffer für einen kompletten Würfel-Frame:
        # Header (Panel, Zeile) wird nur einmal geschrieben.
        self._frame_pkt = np.empty((self.panel_num, self.num_rows, self.row_bytes), dtype=np.uint8)
        self._frame_pkt[:, :, 0] = (np.arange(self.panel_num, dtype=np.uint8) + 1)[:, None]
        self._frame_pkt[:, :, 1] = np.arange(self.num_rows, dtype=np.uint8)[None, :]
        self._frame_payload = self._frame_pkt[:, :, self.header_size:].reshape(
            self.panel_num, self.num_rows, self.num_cols, 3)

        self._panel_pkt = np.empty((self.num_rows, self.row_bytes), dtype=np.uint8)
        self._panel_pkt[:, 1] = np.arange(self.num_rows, dtype=np.uint8)
        self._panel_payload = self._panel_pkt[:, self.header_size:].reshape(
            self.num_rows, self.num_cols, 3)

        self._row_pkt = np.empty(self.row_bytes, dtype=np.uint8)
        self._row_payload = self._row_pkt[self.header_size:].reshape(self.num_cols, 3)

    # ------------------------------------------------------------------
    def _send_rows(self, pkt):
        """Sendet ein (N, 194)-Array zeilenweise als einzelne UDP-Pakete."""
        mv = memoryview(pkt.tobytes())
        n = self.row_bytes
        send = self.s.send
        for off in range(0, len(mv), n):
            send(mv[off:off + n])

    # ------------------------------------------------------------------
    def sendFrame(self, frame):
        """Ganzer Würfel auf einmal.

        frame: uint8-Array der Form (6, 64, 64, 3), Indizes [panel, x, y, rgb]
        """
        # [panel, x, y, rgb] -> [panel, y, x, bgr]
        self._frame_payload[:] = frame.transpose(0, 2, 1, 3)[..., ::-1]
        self._send_rows(self._frame_pkt.reshape(-1, self.row_bytes))

    def sendFullPanel(self, panel, data):
        """Ein Panel. data: (64, 64, 3), Indizes [x, y, rgb] (wie bisher)."""
        self._panel_pkt[:, 0] = panel + 1
        self._panel_payload[:] = data.transpose(1, 0, 2)[..., ::-1]
        self._send_rows(self._panel_pkt)

    def sendRow(self, panel, y, data):
        """Eine Zeile. data: (64, 3), Indizes [x, rgb] (wie bisher)."""
        self._row_pkt[0] = panel + 1
        self._row_pkt[1] = y
        self._row_payload[:] = data[:, ::-1]
        self.s.send(self._row_pkt.tobytes())

    # Im Original identisch zu sendRow – aus Kompatibilitätsgründen beibehalten
    sendPixel = sendRow
