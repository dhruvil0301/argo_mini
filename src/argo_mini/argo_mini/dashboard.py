import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Int32
import json
import threading
import math
import asyncio
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from bleak import BleakClient
from admin_panel import handle_admin_get, handle_admin_post, is_admin_route, admin_router

# ==================== BMS CONFIGURATION ====================
BMS_ADDRESS = "A5:C2:37:2A:22:EC"
RX_CHAR = "0000ff01-0000-1000-8000-00805f9b34fb"
TX_CHAR = "0000ff02-0000-1000-8000-00805f9b34fb"

# Global storage for BMS data shared between threads
latest_bms_data = {
    "voltage": 0.0,
    "current": 0.0,
    "remaining_capacity": 0.0,
    "full_capacity": 0.0,
    "battery_percent": 0.0,
    "temperatures": [],
    "connected": False
}

# Centralized Waypoints File path for sync
WAYPOINTS_FILE = os.path.expanduser('~/argo_mini_ws/src/argo_mini/waypoints/waypoints.json')

dashboard_node = None


def waypoint_name_to_index(target: str) -> int:
    """Map NLP waypoint names (table_1, docking_station) to dashboard indices."""
    if not target:
        return 0
    name = str(target).lower().strip()
    if name in {"docking_station", "home", "dock", "station", "base"}:
        return 0
    if name.startswith("table_"):
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0
    if name.startswith("table") and name[5:].isdigit():
        return int(name[5:])
    return 0


def apply_agent_robot_command(payload: dict):
    """Execute navigation commands emitted by the NLP agent."""
    if not dashboard_node or payload.get("command") != "NAVIGATE":
        return
    target = payload.get("parameters", {}).get("target_waypoint", "") or payload.get("target", "")
    wp_id = waypoint_name_to_index(target)
    msg = Int32()
    msg.data = wp_id
    dashboard_node.cmd_pub.publish(msg)
    dashboard_node.target_destination = f"Table {wp_id}" if wp_id > 0 else "HOME"
    dashboard_node.robot_status = "Navigating"
    print(f"[NLP] Navigation dispatched → waypoint {wp_id} ('{target}')")


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        
        self.robot_status = "Idle"
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.battery_level = 85.0
        self.connection_status = "Connected"
        self.target_destination = "Base Station"
        self.distance_to_goal = 0.0
        self.motor_status = "OK"
        self.lidar_status = "Active"
        self.imu_status = "Calibrated"

        self.latest_scan = None
        self.map_data = {
            "info": {"resolution": 0.05, "width": 0, "height": 0, "originX": 0.0, "originY": 0.0},
            "obstacles": []
        }

        # Master waypoints list synced from JSON
        self.waypoints = self.load_waypoints_from_file()

        self.cmd_pub = self.create_publisher(Int32, '/dashboard_waypoint_cmd', 10)
        
        # Navigation Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.battery_sub = self.create_subscription(BatteryState, '/battery_state', self.battery_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
        
        # SLAM-friendly QoS for /pose and /map
        slam_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_cb, slam_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)

        self.get_logger().info("Dashboard Node Started with Integrated BMS")

    def battery_cb(self, msg):
        # Fallback if BMS is not connected
        if not latest_bms_data["connected"]:
            self.battery_level = float(msg.percentage * 100.0) if msg.percentage <= 1.0 else float(msg.percentage)

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def pose_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def scan_cb(self, msg):
        self.latest_scan = msg

    def map_cb(self, msg):
        width = msg.info.width
        height = msg.info.height
        obstacles = []
        skip = 4 if (width * height > 100000) else 2
        for y in range(0, height, skip):
            for x in range(0, width, skip):
                if msg.data[x + y * width] > 50:
                    obstacles.append([x, y])
        self.map_data = {
            "info": {
                "resolution": msg.info.resolution,
                "width": width,
                "height": height,
                "originX": msg.info.origin.position.x,
                "originY": msg.info.origin.position.y
            },
            "obstacles": obstacles
        }

    def load_waypoints_from_file(self):
        """Sync waypoints from the manager's JSON file safely"""
        if os.path.exists(WAYPOINTS_FILE):
            try:
                with open(WAYPOINTS_FILE, 'r') as f:
                    data = json.load(f)
                    ui_waypoints = []
                    
                    # Sort keys safely: 0 first (Home), then others
                    def sort_keys(k):
                        if k.lower() == 'home' or k == '0': return 0
                        try: return int(k)
                        except: return 999
                    
                    sorted_keys = sorted(data.keys(), key=sort_keys)
                    
                    for k in sorted_keys:
                        v = data[k]
                        # Use Name from JSON or generate one
                        name = v.get("name", "HOME" if k == '0' or k.lower() == 'home' else f"Table {k}")
                        ui_waypoints.append({
                            "name": name,
                            "x": float(v["x"]),
                            "y": float(v["y"]),
                            "locked": (k == '0' or k.lower() == 'home')
                        })
                    return ui_waypoints
            except Exception as e:
                self.get_logger().error(f"Error loading waypoints JSON: {e}")
        
        return [{"name": 'Base Station (Home)', "x": 0.0, "y": 0.0, "locked": True}]

    def save_waypoints_to_file(self):
        """Save current UI waypoints back to the master JSON file"""
        try:
            data = {}
            for i, wp in enumerate(self.waypoints):
                key = str(i)
                if wp['name'].lower() in ['home', 'base station (home)']: key = "0"
                data[key] = {
                    "x": float(wp['x']),
                    "y": float(wp['y']),
                    "qz": 0.0,
                    "qw": 1.0
                }
            with open(WAYPOINTS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f"Saved {len(data)} waypoints to {WAYPOINTS_FILE}")
        except Exception as e:
            self.get_logger().error(f"Failed to write waypoints file: {e}")
    def get_state(self):
        """Assemble current robot telemetry for the web dashboard"""
        # Use BMS percentage if available, otherwise use ROS fallback
        using_bms = latest_bms_data["connected"]
        display_battery = latest_bms_data["battery_percent"] if using_bms else self.battery_level
        
        state = {
            "robotStatus": self.robot_status,
            "currentPosition": f"X: {self.current_x:.2f}, Y: {self.current_y:.2f}",
            "x": self.current_x,
            "y": self.current_y,
            "yaw": self.current_yaw,
            "batteryLevel": display_battery,
            "connectionStatus": self.connection_status,
            "targetDestination": self.target_destination,
            "distanceToGoal": self.distance_to_goal,
            "motorStatus": self.motor_status,
            "lidarStatus": self.lidar_status,
            "imuStatus": self.imu_status,
            "mapData": self.map_data,
            "waypoints": self.waypoints,
            "lidarPoints": self.get_scan_points()
        }
        if using_bms:
            state["bms"] = latest_bms_data
        return state

    def get_scan_points(self):
        if not self.latest_scan:
            return []
        scan = self.latest_scan
        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min <= r <= scan.range_max and r > 0.1:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append({"x": round(x, 3), "y": round(y, 3)})
            angle += scan.angle_increment
        return points

class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        # Gateway: forward admin panel routes
        if is_admin_route('GET', self.path):
            status, body = handle_admin_get(self.path)
            self._send_json(status, body)
            return

        if self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            state = dashboard_node.get_state() if dashboard_node else {}
            self.wfile.write(json.dumps(state).encode())

        elif self.path == '/api/bms':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(latest_bms_data).encode())

        elif self.path == '/api/waypoints':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = dashboard_node.waypoints if dashboard_node else []
            self.wfile.write(json.dumps(data).encode())

        elif self.path == '/favicon.ico':
            self.send_response(204) # No content, stops the 404 error
            self.end_headers()

        elif self.path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Serve Static Images (golden_map.png, etc.)
            try:
                filename = os.path.basename(self.path)
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
                with open(file_path, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(f.read())
            except:
                self.send_response(404)
                self.end_headers()

        elif self.path == '/' or self.path == '/dashboard.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                # Get the absolute path to the HTML file
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b'dashboard.html not found')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Gateway: forward admin panel routes
        if is_admin_route('POST', self.path):
            status, body = handle_admin_post(self.path, post_data)
            if self.path == '/api/argo/events':
                try:
                    req = json.loads(post_data.decode('utf-8'))
                    apply_agent_robot_command(req)
                    if req.get("command") == "NAVIGATE":
                        target = req.get("parameters", {}).get("target_waypoint", "") or req.get("target", "")
                        admin_router.nlp.handle_agent_event({
                            "event":   "STATUS_UPDATE",
                            "status":  "NAVIGATING",
                            "waypoint": target,
                        })
                except Exception as e:
                    print(f"[NLP] Event handling error: {e}")
            self._send_json(status, body)
            return

        if self.path == '/api/command':
            # Handle commands (Emergency Stop, Navigation, etc.)
            try:
                req = json.loads(post_data.decode('utf-8'))
                command = req.get('command')
                args = req.get('args', {})
                
                print(f"Received Command: {command} with args: {args}")
                
                if command == 'navigate' and dashboard_node:
                    # New logic: Send the Waypoint Index to the Manager!
                    wp_id = int(args.get('waypoint', 0))
                    msg = Int32()
                    msg.data = wp_id
                    dashboard_node.cmd_pub.publish(msg)
                    dashboard_node.target_destination = f"Table {wp_id}" if wp_id > 0 else "HOME"
                    dashboard_node.robot_status = "Navigating"

                elif command == 'navigate_to' and dashboard_node:
                    # Fallback for direct coordinates from UI
                    pass

                elif command == 'set_initial_pose' and dashboard_node:
                    # Tell SLAM/Nav2 where the robot is
                    msg = PoseWithCovarianceStamped()
                    msg.header.stamp = dashboard_node.get_clock().now().to_msg()
                    msg.header.frame_id = 'map'
                    msg.pose.pose.position.x = float(args.get('x', 0.0))
                    msg.pose.pose.position.y = float(args.get('y', 0.0))
                    msg.pose.pose.orientation.w = 1.0
                    # Standard deviation for initial pose
                    msg.pose.covariance[0] = 0.25 # x
                    msg.pose.covariance[7] = 0.25 # y
                    msg.pose.covariance[35] = 0.06 # yaw
                    dashboard_node.initial_pose_pub.publish(msg)
                    print(f"Set Initial Pose to X:{msg.pose.pose.position.x}, Y:{msg.pose.pose.position.y}")

                elif command == 'emergency_stop' and dashboard_node:
                    dashboard_node.robot_status = "Emergency Stop"
                    # Add your motor stop logic here
            except Exception as e:
                print(f"Command Error: {e}")

        elif self.path == '/api/waypoints':
            # Save waypoints from Mission Planner
            try:
                new_waypoints = json.loads(post_data.decode('utf-8'))
                if dashboard_node:
                    dashboard_node.waypoints = new_waypoints
                    dashboard_node.save_waypoints_to_file()
                    print(f"Updated {len(new_waypoints)} waypoints and saved to JSON.")
            except Exception as e:
                print(f"Failed to save waypoints: {e}")

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "saved"}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args): pass

# ==================== BMS ASYNC LOGIC ====================
bms_packet = bytearray()

def bms_notification_handler(sender, data):
    global bms_packet
    bms_packet.extend(data)

async def bms_telemetry_loop():
    global bms_packet, latest_bms_data
    from bleak import BleakScanner  # Lazy import to ensure it uses the local loop
    print(f"[BMS] Starting active scanner for {BMS_ADDRESS}...")
    while True:
        try:
            # 1. First, search for the device (more reliable on Linux)
            print(f"[BMS] Searching for battery {BMS_ADDRESS}...")
            device = await BleakScanner.find_device_by_address(BMS_ADDRESS, timeout=15.0)
            
            if not device:
                print(f"[BMS] Device not found in scan. Retrying...")
                await asyncio.sleep(5.0)
                continue

            print(f"[BMS] Device found! Attempting connection to {device.name}...")
            
            # 2. Connect using the discovered device object
            async with BleakClient(device, timeout=20.0) as client:
                print(f"[BMS] Bluetooth Connected!")
                await client.start_notify(RX_CHAR, bms_notification_handler)
                cmd = bytes.fromhex("DD A5 03 00 FF FD 77")
                
                while client.is_connected:
                    bms_packet = bytearray()
                    await client.write_gatt_char(TX_CHAR, cmd, response=False)
                    await asyncio.sleep(2.5) # Wait for data streams
                    
                    data = bytes(bms_packet)
                    if len(data) >= 30:
                        try:
                            # Parsing JBD Protocol
                            voltage = int.from_bytes(data[4:6], "big") / 100.0
                            current = int.from_bytes(data[6:8], "big", signed=True) / 100.0
                            rem_cap = int.from_bytes(data[8:10], "big") / 100.0
                            full_cap = int.from_bytes(data[10:12], "big") / 100.0
                            percent = (rem_cap / full_cap * 100.0) if full_cap > 0 else 0.0
                            
                            count = data[26]
                            temps = []
                            offset = 27
                            for _ in range(min(count, 4)):
                                if offset + 2 <= len(data):
                                    temps.append((int.from_bytes(data[offset:offset+2], "big") - 2731) / 10.0)
                                    offset += 2
                            
                            latest_bms_data.update({
                                "voltage": voltage, "current": current,
                                "remaining_capacity": rem_cap, "full_capacity": full_cap,
                                "battery_percent": percent, "temperatures": temps, "connected": True
                            })
                            print(f"[BMS LIVE] SOC: {percent:.1f}% | {voltage:.2f}V | {current:.2f}A")
                        except Exception as parse_error:
                            print(f"[BMS Error] Data parsing failed: {parse_error}")
                    
                    await asyncio.sleep(3.0)
        except Exception as e:
            print(f"[BMS Error] Loop error: {e}")
            latest_bms_data["connected"] = False
            await asyncio.sleep(5.0)

def run_bms_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bms_telemetry_loop())

# ==================== MAIN ====================
def main():
    # Automatically clear Bluetooth cache and restart service at start
    print("[System] Cleaning Bluetooth cache and restarting service...")
    try:
        import subprocess
        # This will require sudo privileges
        subprocess.run(["sudo", "rm", "-rf", "/var/lib/bluetooth/*"], check=True)
        subprocess.run(["sudo", "systemctl", "restart", "bluetooth"], check=True)
        print("[System] Bluetooth service refreshed successfully.")
    except Exception as e:
        print(f"[System Warning] Could not restart Bluetooth (requires sudo): {e}")

    global dashboard_node
    rclpy.init()
    dashboard_node = DashboardNode()
    
    # Thread 1: HTTP UI Server
    http_server = ThreadingHTTPServer(('0.0.0.0', 8080), DashboardHTTPHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    
    # Thread 2: BMS Bluetooth Collector
    bms_thread = threading.Thread(target=run_bms_thread, daemon=True)
    bms_thread.start()
    
    print("✅ Dashboard & BMS Bridge running on http://0.0.0.0:8080")
    
    try:
        rclpy.spin(dashboard_node)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
