import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
import json
import os
import threading

WAYPOINTS_FILE = os.path.expanduser('~/argo_mini_ws/src/argo_mini/waypoints/waypoints.json')

class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_cb, qos)
        
        self.click_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.click_cb, 10)

        # Dashboard Command Subscriber (Key Integration)
        self.dashboard_sub = self.create_subscription(
            String, '/dashboard_waypoint_cmd', self.dashboard_cmd_cb, 10)

        # Navigation Action Client
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.current_pose = None
        self.clicked_point = None
        self.waypoints = self.load_waypoints()

        self.get_logger().info('Waypoint Manager initialized.')
        self.print_menu()

    def pose_cb(self, msg):
        self.current_pose = msg.pose.pose

    def click_cb(self, msg):
        self.clicked_point = msg.point
        print(f'\n[RViz Click] x={msg.point.x:.3f} y={msg.point.y:.3f}')
        print('> ', end='', flush=True)

    def dashboard_cmd_cb(self, msg):
        """Handle commands coming from the web dashboard"""
        raw = str(msg.data).strip().lower()
        if not raw:
            return
        compact = raw.replace(' ', '')
        action = compact[0]
        try:
            wp_id = int(compact[1:])
        except ValueError:
            self.get_logger().warning(f"Invalid dashboard command: {raw}")
            return

        if action == 'c':
            self.get_logger().info(f"Dashboard requested save command c {wp_id}")
            threading.Thread(target=self.save_current_as, args=(wp_id,), daemon=True).start()
        elif action == 'g':
            self.get_logger().info(f"Dashboard requested goal command g {wp_id}")
            threading.Thread(target=self.go_to, args=(wp_id,), daemon=True).start()
        else:
            self.get_logger().warning(f"Unknown dashboard command prefix: {raw}")

    def load_waypoints(self):
        if os.path.exists(WAYPOINTS_FILE):
            try:
                with open(WAYPOINTS_FILE, 'r') as f:
                    data = json.load(f)
                    self.get_logger().info(f'Loaded {len(data)} waypoints.')
                    return data
            except Exception as e:
                self.get_logger().error(f'Error loading waypoints: {e}')
        return {}

    def save_waypoints(self):
        os.makedirs(os.path.dirname(WAYPOINTS_FILE), exist_ok=True)
        with open(WAYPOINTS_FILE, 'w') as f:
            json.dump(self.waypoints, f, indent=2)
        print('[Saved] Waypoints updated.')

    def print_menu(self):
        print('\n' + '='*55)
        print('           ARGO MINI WAYPOINT MANAGER')
        print('='*55)
        print('  s <0-9>   Save current robot pose')
        print('  c <0-9>   Save clicked point (RViz)')
        print('  g <0-9>   Go to waypoint')
        print('  l         List waypoints')
        print('  q         Quit')
        print('='*55)
        if self.waypoints:
            def sort_key(item):
                key = str(item[0]).lower()
                if key in {'0', 'c0', 'g0', 'c 0', 'g 0', 'home'}:
                    return 0
                if key.startswith('c ') or key.startswith('g '):
                    try:
                        return int(key.split()[1])
                    except (IndexError, ValueError):
                        return 999
                for prefix in ('c', 'g'):
                    if key.startswith(prefix) and key[1:].isdigit():
                        return int(key[1:])
                if key.isdigit():
                    return int(key)
                return 999

            for k, v in sorted(self.waypoints.items(), key=sort_key):
                clean = str(k).lower()
                if clean in {'0', 'c0', 'g0', 'c 0', 'g 0', 'home'}:
                    idx = 0
                elif clean.startswith('c ') or clean.startswith('g '):
                    idx = int(clean.split()[1])
                else:
                    idx = int(clean[1:] if clean[:1] in {'c', 'g'} else clean)
                label = 'HOME' if idx == 0 else f'Table {idx}'
                print(f'  {label}: x={float(v["x"]):.3f}  y={float(v["y"]):.3f}')
        print()

    def save_current_as(self, n):
        if not self.current_pose:
            print("ERROR: No current pose available. Set 2D Pose Estimate in RViz.")
            return
        key = f"c {n}"
        self.waypoints[key] = {
            "x": self.current_pose.position.x,
            "y": self.current_pose.position.y,
            "qz": self.current_pose.orientation.z,
            "qw": self.current_pose.orientation.w
        }
        self.save_waypoints()
        print(f"Saved waypoint {key} from current pose.")

    def save_clicked_as(self, n):
        if not self.clicked_point:
            print("ERROR: Click a point in RViz first using 'Publish Point'.")
            return
        key = f"c {n}"
        self.waypoints[key] = {
            "x": self.clicked_point.x,
            "y": self.clicked_point.y,
            "qz": 0.0,
            "qw": 1.0
        }
        self.save_waypoints()
        print(f"Saved waypoint {key} from clicked point.")

    def go_to(self, n):
        key = f"c{n}"
        if key not in self.waypoints:
            print(f"ERROR: Waypoint {n} not found.")
            return

        wp = self.waypoints[key]
        print(f"\n[NAV] Sending goal to waypoint g{n} → x={wp['x']:.3f}, y={wp['y']:.3f}")

        if not self.nav_client.wait_for_server(timeout_sec=8.0):
            print("ERROR: Nav2 action server not available!")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(wp['x'])
        goal.pose.pose.position.y = float(wp['y'])
        goal.pose.pose.orientation.z = float(wp.get('qz', 0.0))
        goal.pose.pose.orientation.w = float(wp.get('qw', 1.0))

        future = self.nav_client.send_goal_async(goal, feedback_callback=self.feedback_cb)
        future.add_done_callback(lambda f: self.goal_response_cb(f, n))

    def feedback_cb(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        print(f"\r[Progress] Distance remaining: {dist:.2f} m", end='', flush=True)

    def goal_response_cb(self, future, n):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print(f"\n[FAIL] Goal to waypoint g{n} was REJECTED by Nav2.")
            return
        print(f"\n[ACCEPTED] Moving toward waypoint g{n}...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.result_cb(f, n))

    def result_cb(self, future, n):
        result = future.result()
        if result.status == 4:  # Succeeded
            print(f"\n[SUCCESS] Reached waypoint g{n}!")
        else:
            print(f"\n[FAIL] Failed to reach waypoint g{n} (status: {result.status})")

def main():
    rclpy.init()
    node = WaypointManager()
    
    # Run ROS spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            cmd = input('> ').strip().lower()
            if not cmd:
                continue
            parts = cmd.split()
            if parts[0] == 'q':
                break
            elif parts[0] == 'l':
                node.print_menu()
            elif parts[0] == 's' and len(parts) == 2:
                node.save_current_as(int(parts[1]))
            elif parts[0] == 'c' and len(parts) == 2:
                node.save_clicked_as(int(parts[1]))
            elif parts[0] == 'g' and len(parts) == 2:
                node.go_to(int(parts[1]))
            else:
                print("Commands: s <0-9> | c <0-9> | g <0-9> | l | q")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
