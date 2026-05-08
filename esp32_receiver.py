"""
esp32_receiver.py
-----------------
Reads serial data from ESP32, parses sensor values,
runs ML inference, and broadcasts to dashboard via WebSocket.

Data flow:
ESP32 (Serial 115200) → parse → ml_model.py → WebSocket → dashboard.html
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
import asyncio
import websockets
import logging
from datetime import datetime

from ml_model import TremorMLModel


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger(__name__)


# SETTINGS
BAUD_RATE = 115200
HOST = "localhost"
PORT = 8765


# GLOBAL DATA
latest_data = {
    "tremor_pct": 0.0,
    "hr_bpm": 0.0,
    "hr_severity": 0.0,
    "combined_pct": 0.0,
    "led_state": "GREEN",
    "ml_severity": 0.0,
    "ml_label": "Normal",
    "timestamp": "",
    "connected": False,
}

data_lock = threading.Lock()
connected_clients = set()


# FIND ESP32 PORT
def find_port():

    ports = serial.tools.list_ports.comports()

    for p in ports:

        name = (p.description or "") + (p.manufacturer or "")

        if any(x in name for x in ["CP210", "CH340", "USB"]):

            return p.device

    if ports:
        return ports[0].device

    return None


# PARSE SERIAL LINE
def parse_line(line):

    if "|" not in line:
        return None

    parts = [p.strip().replace("%", "") for p in line.split("|")]

    if len(parts) < 5:
        return None

    try:

        return {
            "tremor_pct": float(parts[0]),
            "hr_bpm": float(parts[1]),
            "hr_severity": float(parts[2]),
            "combined_pct": float(parts[3]),
            "led_state": "RED" if "RED" in parts[4].upper() else "GREEN"
        }

    except:
        return None


ml_model = TremorMLModel()


# SERIAL THREAD
def serial_thread():

    global latest_data

    while True:

        port = find_port()

        if not port:
            log.warning("ESP32 not found")
            time.sleep(3)
            continue

        try:

            with serial.Serial(port, BAUD_RATE, timeout=2) as ser:

                log.info(f"Connected to ESP32 on {port}")

                while True:

                    raw = ser.readline()

                    if not raw:
                        continue

                    try:
                        line = raw.decode()
                    except:
                        continue

                    parsed = parse_line(line)

                    if parsed is None:
                        continue


                    ml_result = ml_model.predict(
                        parsed["tremor_pct"],
                        parsed["hr_bpm"],
                        parsed["combined_pct"]
                    )


                    with data_lock:

                        latest_data.update(parsed)

                        latest_data["ml_severity"] = ml_result["severity"]

                        latest_data["ml_label"] = ml_result["label"]

                        latest_data["timestamp"] = datetime.now().strftime("%H:%M:%S")

                        latest_data["connected"] = True


                    log.info(
                        f"Tremor {parsed['tremor_pct']:.1f}% | "
                        f"HR {parsed['hr_bpm']:.1f} | "
                        f"{ml_result['label']} {ml_result['severity']:.1f}%"
                    )


        except Exception as e:

            log.warning(e)

            time.sleep(3)



# WEBSOCKET HANDLER
async def handler(ws):

    connected_clients.add(ws)

    log.info("Dashboard connected")

    try:

        async for _ in ws:
            pass

    finally:

        connected_clients.remove(ws)

        log.info("Dashboard disconnected")



# SEND DATA TO DASHBOARD
async def send_loop():

    global connected_clients

    while True:

        await asyncio.sleep(0.2)

        if len(connected_clients) == 0:
            continue

        with data_lock:

            msg = json.dumps(latest_data)

        dead = set()

        for ws in connected_clients.copy():

            try:

                await ws.send(msg)

            except:

                dead.add(ws)

        connected_clients -= dead



# START SERVER
async def main():

    async with websockets.serve(handler, HOST, PORT):

        log.info(f"WebSocket started ws://{HOST}:{PORT}")

        await send_loop()



# RUN
if __name__ == "__main__":

    t = threading.Thread(target=serial_thread, daemon=True)

    t.start()

    asyncio.run(main())