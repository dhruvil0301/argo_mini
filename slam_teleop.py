#!/usr/bin/env python3
"""
Argo Mini SLAM teleop — forward + pivot only, no reverse.

  w        forward
  a        pivot left  (left stops, right runs)
  d        pivot right (right stops, left runs)
  s        stop
  space    stop

  Ctrl+C   quit
"""

import sys
import tty
import termios
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LIN_VEL = 0.15   # m/s  — keep slow for good SLAM
ANG_VEL = 0.50   # rad/s

KEYS = {
    'w': ( LIN_VEL,  0.0),
    'a': ( 0.0,      ANG_VEL),
    'd': ( 0.0,     -ANG_VEL),
    's': ( 0.0,      0.0),
    ' ': ( 0.0,      0.0),
}

BANNER = """
=========================================
  ARGO MINI — SLAM TELEOP (no reverse)
=========================================
  w  → forward
  a  → pivot left
  d  → pivot right
  s / space → stop
  Ctrl+C → quit
=========================================
"""

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    rclpy.init()
    node = Node('slam_teleop')
    pub  = node.create_publisher(Twist, '/cmd_vel', 10)

    settings = termios.tcgetattr(sys.stdin)
    print(BANNER)

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key == '\x03':   # Ctrl+C
                break

            lin, ang = KEYS.get(key, (None, None))
            if lin is None:
                continue

            msg = Twist()
            msg.linear.x  = lin
            msg.angular.z = ang
            pub.publish(msg)

            label = {
                'w': 'FORWARD',
                'a': 'PIVOT LEFT',
                'd': 'PIVOT RIGHT',
                's': 'STOP',
                ' ': 'STOP',
            }.get(key, '')
            print(f'\r  {label:<14}  lin={lin:+.2f}  ang={ang:+.2f}   ', end='', flush=True)

    finally:
        # Send stop on exit
        stop = Twist()
        pub.publish(stop)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print('\n[teleop] stopped.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
