#!/usr/bin/env python3
# Voice-to-Grasp - PC side of the glove controller (paper Section 2.1).
#
# Sends two set-points in percent (channel 1 = thumb and index finger, channel 2 = other three
# fingers) as one text line "p1 p2\n" over USB serial (115200 baud) to the Arduino running
# gloves_control/gloves_control.ino, which drives the two electronic pressure regulators.
# Used by GuiUserStudyNew.py. Can also be run on its own to test the glove:
#     python two_regulator_controller.py        (set PORT below to your Arduino's port)


import serial
import sys
import time

# CONFIGURATION
PORT = 'COM5'           # Change to your Arduino port
BAUD = 115200
MIN_PERCENT = 0
MAX_PERCENT = 100

class TwoRegulatorController:

    def __init__(self, port='COM5', baud=115200):
        try:
            self.ser = serial.Serial(
                port,
                baudrate=baud,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                write_timeout=1
            )
            time.sleep(2)  # allow Arduino reset
            print("Connected to Arduino on", PORT)
        except Exception as e:
            print("Failed to connect:", e)
            sys.exit(1)

    def send_percents(self, p1, p2):
        """
        Sends two percentage values to Arduino in the format:
        <p1,p2>
        where each is clamped 0–100
        """
        # Clamp to valid range
        p1 = max(MIN_PERCENT, min(MAX_PERCENT, p1))
        p2 = max(MIN_PERCENT, min(MAX_PERCENT, p2))

        frame = f"{p1} {p2}\n"
        self.ser.write(frame.encode("utf-8"))
        self.ser.flush()

        # Optional: read Arduino reply
        if self.ser.in_waiting:
            resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
            print("Arduino:", resp)

    def close(self):
        self.ser.close()


# if __name__ == "__main__":
#     ctrl = TwoRegulatorController()
#     try:
#         while True:
#             try:
#                 raw = input("Enter two percents (e.g. 50 75), or 'q' to quit: ").strip()
#                 if raw.lower() == 'q':
#                     break
#                 parts = raw.replace(',', ' ').split()
#                 if len(parts) != 2:
#                     print("Please enter exactly two values")
#                     continue
#                 p1, p2 = float(parts[0]), float(parts[1])
#                 ctrl.send_percents(p1, p2)
#             except ValueError:
#                 print("Invalid input, enter numbers 0–100")
#     except KeyboardInterrupt:
#         pass
#     finally:
#         ctrl.close()
#         print("Disconnected")
