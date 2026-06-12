#!/usr/bin/env python3
"""Minimal depth-camera obstacle detection and ESP32 motor test.

This script is intentionally standalone and does not require ROS2.
It reads a depth camera, checks the center obstacle distance,
and sends a stop command to the ESP32 motor controller when an
obstacle is closer than the configured threshold.
"""

import argparse
import sys
import time

import numpy as np
import serial
import cv2


def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        time.sleep(1.0)
        ser.reset_input_buffer()
        print(f'Opened serial port: {port} @ {baud}')
        return ser
    except Exception as exc:
        print(f'ERROR: cannot open serial port {port}: {exc}')
        return None


def open_depth_camera(index):
    # Try OpenNI depth capture first, then fallback to a normal camera.
    for backend in (cv2.CAP_OPENNI2, cv2.CAP_OPENNI, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f'Opened camera index {index} using backend {backend}')
            return cap, backend
    return None, None


def read_depth_frame(cap, backend):
    if backend in (cv2.CAP_OPENNI2, cv2.CAP_OPENNI):
        if not cap.grab():
            return None
        ok, depth = cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
        if not ok or depth is None:
            return None
        return depth

    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # Fallback: no real depth available.
    return None


def send_motor_command(ser, left, right):
    if ser is None:
        return
    line = f'V {left} {right}\n'.encode('utf-8')
    try:
        ser.write(line)
        ser.flush()
    except Exception as exc:
        print(f'ERROR: failed to send motor command: {exc}')


def compute_center_distance(depth, threshold_m, roi_ratio=0.4):
    if depth is None:
        return None
    height, width = depth.shape[:2]
    cx = width // 2
    cy = height // 2
    rw = max(1, int(width * roi_ratio // 2))
    rh = max(1, int(height * roi_ratio // 2))
    roi = depth[cy - rh:cy + rh, cx - rw:cx + rw]
    valid = roi[(roi > 0) & (roi < 10000)]
    if valid.size == 0:
        return None
    min_mm = int(np.min(valid))
    median_mm = int(np.median(valid))
    return min_mm / 1000.0, median_mm / 1000.0, roi


def draw_depth_overlay(frame, dist_text, threshold_m, min_dist):
    if frame is None:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, 70), (0, 0, 0), -1)
    cv2.putText(frame, dist_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if min_dist is None or min_dist > threshold_m else (0, 0, 255), 2)
    cv2.putText(frame, f'threshold {threshold_m:.2f} m', (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def make_depth_visual(depth):
    if depth is None:
        return None
    depth_vis = np.uint8(np.clip(depth / 50.0, 0, 255))
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def main():
    parser = argparse.ArgumentParser(
        description='Standalone depth-camera obstacle detection test for ESP32 motors.')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0',
                        help='ESP32 serial port for motor commands')
    parser.add_argument('--baud', default=115200, type=int,
                        help='ESP32 serial speed')
    parser.add_argument('--camera-index', default=0, type=int,
                        help='Depth camera index')
    parser.add_argument('--threshold', default=0.8, type=float,
                        help='Stop threshold in meters')
    parser.add_argument('--forward-left', default=107, type=int,
                        help='Forward left DAC value')
    parser.add_argument('--forward-right', default=106, type=int,
                        help='Forward right DAC value')
    parser.add_argument('--stop-left', default=0, type=int,
                        help='Stop left DAC value')
    parser.add_argument('--stop-right', default=0, type=int,
                        help='Stop right DAC value')
    parser.add_argument('--display', action='store_true',
                        help='Show camera image and depth overlay')
    args = parser.parse_args()

    ser = open_serial(args.serial_port, args.baud)
    cap, backend = open_depth_camera(args.camera_index)
    if cap is None:
        print('ERROR: cannot open camera. Exiting.')
        sys.exit(1)

    use_depth = backend in (cv2.CAP_OPENNI2, cv2.CAP_OPENNI)
    print('Automatic obstacle avoidance enabled.')
    print('Depth mode:' if use_depth else 'No depth mode. Cannot use real depth data.')

    try:
        while True:
            depth = read_depth_frame(cap, backend)
            if depth is None:
                if use_depth:
                    print('WARNING: depth frame not available from depth camera.')
                min_dist = None
                median_dist = None
            else:
                result = compute_center_distance(depth, args.threshold)
                if result is None:
                    min_dist = None
                    median_dist = None
                else:
                    min_dist, median_dist, roi = result

            if min_dist is not None and min_dist < args.threshold:
                send_motor_command(ser, args.stop_left, args.stop_right)
                state = 'STOPPED'
            elif min_dist is None and use_depth:
                send_motor_command(ser, args.stop_left, args.stop_right)
                state = 'STOPPED'
            else:
                send_motor_command(ser, args.forward_left, args.forward_right)
                state = 'MOVING'

            dist_text = 'No depth' if min_dist is None else f'min {min_dist:.2f} m, median {median_dist:.2f} m'
            print(f'[{state}] {dist_text}')

            if args.display:
                frame = None
                if depth is not None:
                    frame = make_depth_visual(depth)
                if frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_depth_overlay(frame, dist_text, args.threshold, min_dist)
                cv2.imshow('Depth Obstacle Test', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print('Interrupted by user.')
    finally:
        send_motor_command(ser, args.stop_left, args.stop_right)
        if ser is not None:
            ser.close()
        if cap is not None:
            cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
