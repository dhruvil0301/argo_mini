#!/usr/bin/env python3
"""Minimal YDLidar obstacle detection and ESP32 motor test.

This script is standalone and does not require ROS2.
It uses a 2D lidar scan to detect obstacles in front of the robot
and sends stop/forward commands over serial to the ESP32.

Note: YDLidar is a 2D laser scanner, not a depth camera.
This file is separate from depth_obstacle_test.py.
"""

import argparse
import sys
import time

import numpy as np
import serial

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None


def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        time.sleep(1.0)
        ser.reset_input_buffer()
        print(f'Opened ESP32 serial port: {port} @ {baud}')
        return ser
    except Exception as exc:
        print(f'ERROR: cannot open serial port {port}: {exc}')
        return None


def open_lidar(port):
    if RPLidar is None:
        print('ERROR: rplidar package is not installed.')
        print('Install it with: python3 -m pip install rplidar')
        return None
    try:
        lidar = RPLidar(port, baudrate=115200, timeout=3)
        print(f'Opened lidar port: {port}')
        return lidar
    except Exception as exc:
        print(f'ERROR: cannot open lidar port {port}: {exc}')
        return None


def stop_lidar(lidar):
    if lidar is None:
        return
    try:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    except Exception:
        pass


def compute_forward_distance(scan, forward_width=60):
    if not scan:
        return None
    angles = np.array([point[1] for point in scan], dtype=float)
    distances = np.array([point[2] for point in scan], dtype=float) / 1000.0
    valid = (distances > 0) & (distances < 20.0)
    angles = angles[valid]
    distances = distances[valid]
    if angles.size == 0:
        return None
    # Normalize to [0, 360)
    angles = np.mod(angles, 360.0)
    forward_mask = ((angles <= forward_width / 2.0) | (angles >= 360.0 - forward_width / 2.0))
    forward_distances = distances[forward_mask]
    if forward_distances.size == 0:
        return None
    return float(np.min(forward_distances)), float(np.median(forward_distances))


def read_lidar_scan(lidar, timeout=5):
    if lidar is None:
        return None
    try:
        scan_generator = lidar.iter_scans(max_buf_meas=500)
        start = time.time()
        while time.time() - start < timeout:
            scan = next(scan_generator, None)
            if scan:
                return scan
    except StopIteration:
        return None
    except Exception as exc:
        print(f'ERROR: failed to read lidar scan: {exc}')
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


def main():
    parser = argparse.ArgumentParser(
        description='Standalone YDLidar obstacle detection test for ESP32 motors.')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0',
                        help='ESP32 serial port for motor commands')
    parser.add_argument('--baud', default=115200, type=int,
                        help='ESP32 serial speed')
    parser.add_argument('--lidar-port', default='/dev/ttyUSB1',
                        help='YDLidar serial port')
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
    parser.add_argument('--forward-width', default=60, type=float,
                        help='Forward obstacle sector width in degrees')
    args = parser.parse_args()

    ser = open_serial(args.serial_port, args.baud)
    lidar = open_lidar(args.lidar_port)
    if lidar is None:
        sys.exit(1)

    print('Automatic obstacle avoidance enabled for 2D lidar.')
    try:
        while True:
            scan = read_lidar_scan(lidar)
            if scan is None:
                print('WARNING: no lidar scan received; stopping motors.')
                send_motor_command(ser, args.stop_left, args.stop_right)
                time.sleep(0.2)
                continue

            result = compute_forward_distance(scan, args.forward_width)
            if result is None:
                print('WARNING: no valid forward distance; stopping motors.')
                send_motor_command(ser, args.stop_left, args.stop_right)
                continue

            min_dist, median_dist = result
            if min_dist < args.threshold:
                send_motor_command(ser, args.stop_left, args.stop_right)
                state = 'STOPPED'
            else:
                send_motor_command(ser, args.forward_left, args.forward_right)
                state = 'MOVING'

            print(f'[{state}] min {min_dist:.2f} m, median {median_dist:.2f} m')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('Interrupted by user.')
    finally:
        send_motor_command(ser, args.stop_left, args.stop_right)
        if ser is not None:
            ser.close()
        stop_lidar(lidar)


if __name__ == '__main__':
    main()
