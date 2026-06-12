#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tkinter as tk
from tkinter import ttk
import threading

class WaypointUINode(Node):
    def __init__(self):
        super().__init__('waypoint_ui')
        
        # Waypoint definitions
        self.waypoints = {
            0: {"name": "Base Station", "x": 0.0, "y": 0.0},
            1: {"name": "Table 1", "x": -1.5, "y": 1.5},
            2: {"name": "Table 2", "x": 1.5, "y": 1.5},
            3: {"name": "Table 3", "x": -1.5, "y": -1.5},
            4: {"name": "Table 4", "x": 1.5, "y": -1.5},
        }
        
        # Publisher
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.get_logger().info('Waypoint UI Node Started')
    
    def send_goal(self, waypoint_id):
        """Send navigation goal to waypoint"""
        if waypoint_id not in self.waypoints:
            self.get_logger().error(f"Invalid waypoint: {waypoint_id}")
            return
        
        wp = self.waypoints[waypoint_id]
        
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(wp['x'])
        goal.pose.position.y = float(wp['y'])
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        self.get_logger().info(f"Sent goal: {wp['name']}")

class WaypointUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Argo Mini - Waypoint Manager")
        self.root.geometry("600x500")
        self.root.configure(bg="#0C447C")
        
        # Configure style
        self.root.option_add("*Font", "Segoe 12")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create UI widgets"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#185FA5", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="Argo Mini Waypoint Manager", 
                        font=("Segoe", 20, "bold"), bg="#185FA5", fg="white")
        title.pack(pady=10)
        
        subtitle = tk.Label(header_frame, text="Select destination for autonomous delivery", 
                           font=("Segoe", 11), bg="#185FA5", fg="white", wraplength=500)
        subtitle.pack(pady=5)
        
        # Content frame
        content_frame = tk.Frame(self.root, bg="#0C447C")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Status display
        status_label = tk.Label(content_frame, text="Current Goal:", 
                               font=("Segoe", 12, "bold"), bg="#0C447C", fg="white")
        status_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.status_var = tk.StringVar(value="None")
        status_display = tk.Label(content_frame, textvariable=self.status_var,
                                 font=("Segoe", 16, "bold"), bg="#E6F1FB", 
                                 fg="#185FA5", pady=15, relief=tk.FLAT)
        status_display.pack(fill=tk.X, pady=(0, 20))
        
        # Waypoint buttons frame
        buttons_frame = tk.Frame(content_frame, bg="#0C447C")
        buttons_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Table buttons (2x2 grid)
        tables_label = tk.Label(buttons_frame, text="Delivery Locations", 
                               font=("Segoe", 12, "bold"), bg="#0C447C", fg="white")
        tables_label.pack(anchor=tk.W, pady=(0, 10))
        
        table_grid = tk.Frame(buttons_frame, bg="#0C447C")
        table_grid.pack(fill=tk.X, pady=10)
        
        # Table 1
        btn1 = tk.Button(table_grid, text="Table 1", font=("Segoe", 14, "bold"),
                        bg="#185FA5", fg="white", padx=20, pady=15,
                        command=lambda: self._button_click(1, "Table 1"))
        btn1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Table 2
        btn2 = tk.Button(table_grid, text="Table 2", font=("Segoe", 14, "bold"),
                        bg="#185FA5", fg="white", padx=20, pady=15,
                        command=lambda: self._button_click(2, "Table 2"))
        btn2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Table 3
        btn3 = tk.Button(table_grid, text="Table 3", font=("Segoe", 14, "bold"),
                        bg="#185FA5", fg="white", padx=20, pady=15,
                        command=lambda: self._button_click(3, "Table 3"))
        btn3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Table 4
        btn4 = tk.Button(table_grid, text="Table 4", font=("Segoe", 14, "bold"),
                        bg="#185FA5", fg="white", padx=20, pady=15,
                        command=lambda: self._button_click(4, "Table 4"))
        btn4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights
        table_grid.grid_columnconfigure(0, weight=1)
        table_grid.grid_columnconfigure(1, weight=1)
        
        # Base station button
        base_label = tk.Label(buttons_frame, text="Return to Base", 
                             font=("Segoe", 12, "bold"), bg="#0C447C", fg="white")
        base_label.pack(anchor=tk.W, pady=(20, 10))
        
        base_btn = tk.Button(buttons_frame, text="Base Station (Home)", 
                            font=("Segoe", 14, "bold"),
                            bg="#27ae60", fg="white", padx=20, pady=15,
                            command=lambda: self._button_click(0, "Base Station"))
        base_btn.pack(fill=tk.X, pady=5)
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg="#0C447C")
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        cancel_btn = tk.Button(control_frame, text="Cancel Navigation", 
                              font=("Segoe", 12, "bold"),
                              bg="#A32D2D", fg="white", padx=20, pady=10,
                              command=self._cancel_navigation)
        cancel_btn.pack(fill=tk.X, pady=5)
        
        exit_btn = tk.Button(control_frame, text="Exit", 
                            font=("Segoe", 12, "bold"),
                            bg="#95a5a6", fg="white", padx=20, pady=10,
                            command=self._exit)
        exit_btn.pack(fill=tk.X, pady=5)
    
    def _button_click(self, waypoint_id, name):
        """Handle waypoint button click"""
        self.status_var.set(name)
        self.node.send_goal(waypoint_id)
    
    def _cancel_navigation(self):
        """Cancel current navigation"""
        self.status_var.set("Navigation Cancelled")
        self.node.get_logger().info("Navigation cancelled")
    
    def _exit(self):
        """Exit application"""
        self.root.quit()
    
    def run(self):
        """Run the UI"""
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointUINode()
    
    # Create UI in main thread
    ui = WaypointUI(node)
    
    # Spin ROS in background thread
    def spin_ros():
        rclpy.spin(node)
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    # Run UI (blocks until window closes)
    ui.run()
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
