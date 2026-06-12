#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import tkinter as tk
from tkinter import messagebox
import threading
import math

class PoseSetterNode(Node):
    def __init__(self):
        super().__init__('pose_setter')
        
        # Publisher for initial pose
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )
        
        # Subscriber to listen for 2D pose estimate from RViz
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.pose_callback,
            10
        )
        
        self.last_pose = None
        self.get_logger().info('Pose Setter Node Started - Use RViz 2D Pose Estimate tool')
    
    def pose_callback(self, msg):
        """Capture pose from RViz"""
        self.last_pose = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z_quat = msg.pose.pose.orientation.z
        w_quat = msg.pose.pose.orientation.w
        
        yaw = math.atan2(2.0 * (w_quat * z_quat), 1.0 - 2.0 * (z_quat * z_quat))
        yaw_deg = math.degrees(yaw)
        
        self.get_logger().info(f"Pose received - X: {x:.3f}, Y: {y:.3f}, Yaw: {yaw_deg:.1f}°")

class PoseSetterUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Argo Mini - Interactive Waypoint Setter")
        self.root.geometry("900x700")
        self.root.configure(bg="#0C447C")
        
        self.waypoints = {}
        self.canvas_width = 800
        self.canvas_height = 600
        self.scale = 50  # pixels per meter
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create UI widgets"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#185FA5", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="Interactive Waypoint Setter", 
                        font=("Segoe", 18, "bold"), bg="#185FA5", fg="white")
        title.pack(pady=10)
        
        subtitle = tk.Label(header_frame, text="Click on map in RViz ? Waypoint appears here", 
                           font=("Segoe", 10), bg="#185FA5", fg="white")
        subtitle.pack(pady=5)
        
        # Content frame
        content_frame = tk.Frame(self.root, bg="#0C447C")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Instructions
        instr_label = tk.Label(content_frame, text="Instructions:", 
                              font=("Segoe", 11, "bold"), bg="#0C447C", fg="white")
        instr_label.pack(anchor=tk.W, pady=(0, 5))
        
        instr_text = tk.Label(content_frame, 
            text="1. Click '2D Pose Estimate' in RViz\n2. Click on map at waypoint location\n3. Click button to assign waypoint\n4. Export code when done",
            font=("Segoe", 10), bg="#0C447C", fg="white", justify=tk.LEFT)
        instr_text.pack(anchor=tk.W, pady=(0, 15))
        
        # Canvas for map visualization
        canvas_label = tk.Label(content_frame, text="Map View (Robot's perspective):", 
                               font=("Segoe", 11, "bold"), bg="#0C447C", fg="white")
        canvas_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.canvas = tk.Canvas(content_frame, width=self.canvas_width, height=self.canvas_height,
                               bg="#E6F1FB", highlightthickness=2, highlightbackground="#185FA5")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Draw grid
        self._draw_grid()
        
        # Waypoint buttons frame
        buttons_frame = tk.Frame(content_frame, bg="#0C447C")
        buttons_frame.pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(buttons_frame, text="Assign Last Click to:", 
                font=("Segoe", 10, "bold"), bg="#0C447C", fg="white").pack(side=tk.LEFT, padx=(0, 10))
        
        # Waypoint selection buttons
        btn_base = tk.Button(buttons_frame, text="Base (0)", font=("Segoe", 9, "bold"),
                            bg="#27ae60", fg="white", padx=10, pady=8,
                            command=lambda: self._assign_waypoint(0, "Base Station"))
        btn_base.pack(side=tk.LEFT, padx=3)
        
        btn1 = tk.Button(buttons_frame, text="Table 1", font=("Segoe", 9, "bold"),
                        bg="#185FA5", fg="white", padx=10, pady=8,
                        command=lambda: self._assign_waypoint(1, "Table 1"))
        btn1.pack(side=tk.LEFT, padx=3)
        
        btn2 = tk.Button(buttons_frame, text="Table 2", font=("Segoe", 9, "bold"),
                        bg="#185FA5", fg="white", padx=10, pady=8,
                        command=lambda: self._assign_waypoint(2, "Table 2"))
        btn2.pack(side=tk.LEFT, padx=3)
        
        btn3 = tk.Button(buttons_frame, text="Table 3", font=("Segoe", 9, "bold"),
                        bg="#185FA5", fg="white", padx=10, pady=8,
                        command=lambda: self._assign_waypoint(3, "Table 3"))
        btn3.pack(side=tk.LEFT, padx=3)
        
        btn4 = tk.Button(buttons_frame, text="Table 4", font=("Segoe", 9, "bold"),
                        bg="#185FA5", fg="white", padx=10, pady=8,
                        command=lambda: self._assign_waypoint(4, "Table 4"))
        btn4.pack(side=tk.LEFT, padx=3)
        
        # Current position display
        pos_label = tk.Label(content_frame, text="Last RViz Click:", 
                            font=("Segoe", 10, "bold"), bg="#0C447C", fg="white")
        pos_label.pack(anchor=tk.W, pady=(10, 5))
        
        pos_frame = tk.Frame(content_frame, bg="#E6F1FB")
        pos_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pos_var = tk.StringVar(value="Waiting for RViz click...")
        pos_display = tk.Label(pos_frame, textvariable=self.pos_var,
                              font=("Segoe", 11), bg="#E6F1FB", fg="#185FA5",
                              pady=10, justify=tk.LEFT)
        pos_display.pack(fill=tk.X, padx=10)
        
        # Saved waypoints
        saved_label = tk.Label(content_frame, text="Saved Waypoints:", 
                              font=("Segoe", 10, "bold"), bg="#0C447C", fg="white")
        saved_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.waypoints_frame = tk.Frame(content_frame, bg="#E6F1FB")
        self.waypoints_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.waypoints_display = tk.Label(self.waypoints_frame, text="No waypoints saved yet",
                                         font=("Segoe", 10), bg="#E6F1FB", fg="#666666",
                                         pady=10, justify=tk.LEFT)
        self.waypoints_display.pack(fill=tk.X, padx=10)
        
        # Export button
        export_frame = tk.Frame(content_frame, bg="#0C447C")
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        export_btn = tk.Button(export_frame, text="Export Python Code (Copy to Clipboard)", 
                              font=("Segoe", 11, "bold"),
                              bg="#185FA5", fg="white", padx=20, pady=10,
                              command=self._export_waypoints)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(export_frame, text="Clear All", 
                             font=("Segoe", 11, "bold"),
                             bg="#A32D2D", fg="white", padx=20, pady=10,
                             command=self._clear_all)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Start refresh timer
        self._start_refresh_timer()
    
    def _draw_grid(self):
        """Draw grid on canvas"""
        self.canvas.delete("grid")
        
        # Draw grid lines
        for i in range(-4, 5):
            x = self.canvas_width // 2 + i * self.scale
            self.canvas.create_line(x, 0, x, self.canvas_height, fill="#ddd", dash=(2, 2), tags="grid")
            self.canvas.create_text(x, self.canvas_height - 10, text=str(i), fill="#999", tags="grid")
        
        for i in range(-3, 4):
            y = self.canvas_height // 2 - i * self.scale
            self.canvas.create_line(0, y, self.canvas_width, y, fill="#ddd", dash=(2, 2), tags="grid")
            self.canvas.create_text(10, y, text=str(i), fill="#999", tags="grid")
        
        # Draw center (origin)
        cx = self.canvas_width // 2
        cy = self.canvas_height // 2
        self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill="#185FA5", outline="#0C447C", width=2, tags="grid")
        self.canvas.create_text(cx+15, cy-10, text="Origin (0,0)", fill="#185FA5", tags="grid")
        
        # Draw existing waypoints
        self._redraw_waypoints()
    
    def _redraw_waypoints(self):
        """Redraw waypoints on canvas"""
        self.canvas.delete("waypoint")
        
        colors = {0: "#27ae60", 1: "#185FA5", 2: "#185FA5", 3: "#185FA5", 4: "#185FA5"}
        
        for wp_id, wp_data in sorted(self.waypoints.items()):
            x = wp_data["x"]
            y = wp_data["y"]
            
            # Convert to canvas coordinates
            cx = self.canvas_width // 2 + x * self.scale
            cy = self.canvas_height // 2 - y * self.scale
            
            # Draw circle
            radius = 8
            self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                                   fill=colors.get(wp_id, "#185FA5"),
                                   outline="#0C447C", width=2, tags="waypoint")
            
            # Draw label
            self.canvas.create_text(cx, cy-20, text=f"WP {wp_id}",
                                   font=("Segoe", 9, "bold"),
                                   fill=colors.get(wp_id, "#185FA5"), tags="waypoint")
    
    def _assign_waypoint(self, wp_id, name):
        """Assign current RViz pose to waypoint"""
        if self.node.last_pose is None:
            messagebox.showerror("Error", "Click 2D Pose Estimate in RViz first!")
            return
        
        x = self.node.last_pose.pose.pose.x
        y = self.node.last_pose.pose.pose.position.y
        z_quat = self.node.last_pose.pose.pose.orientation.z
        w_quat = self.node.last_pose.pose.pose.orientation.w
        
        self.waypoints[wp_id] = {
            "id": wp_id,
            "name": name,
            "x": round(x, 3),
            "y": round(y, 3)
        }
        
        self._redraw_waypoints()
        self._update_waypoints_display()
        messagebox.showinfo("Saved", f"Waypoint {wp_id} ({name}) saved at X={x:.3f}, Y={y:.3f}")
    
    def _update_waypoints_display(self):
        """Update waypoints display"""
        if not self.waypoints:
            text = "No waypoints saved yet"
        else:
            text = ""
            for wp_id in sorted(self.waypoints.keys()):
                wp = self.waypoints[wp_id]
                text += f"WP {wp_id} ({wp['name']}): X={wp['x']:.3f}, Y={wp['y']:.3f}\n"
        
        self.waypoints_display.config(text=text.strip())
    
    def _export_waypoints(self):
        """Export waypoints as Python code"""
        if not self.waypoints:
            messagebox.showwarning("No Waypoints", "Save at least one waypoint first!")
            return
        
        code = "self.waypoints = {\n"
        for wp_id in sorted(self.waypoints.keys()):
            wp = self.waypoints[wp_id]
            code += f'    {wp_id}: {{"name": "{wp["name"]}", "x": {wp["x"]}, "y": {wp["y"]}}},\n'
        code += "}"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        
        messagebox.showinfo("Copied!", 
            f"Code for {len(self.waypoints)} waypoint(s) copied to clipboard!\n\n"
            f"Paste this into waypoint_ui.py or dashboard.py")
    
    def _clear_all(self):
        """Clear all waypoints"""
        if messagebox.askyesno("Confirm", "Clear all waypoints?"):
            self.waypoints.clear()
            self._redraw_waypoints()
            self._update_waypoints_display()
    
    def _start_refresh_timer(self):
        """Auto-refresh position every 200ms"""
        def refresh():
            if self.node.last_pose:
                x = self.node.last_pose.pose.pose.position.x
                y = self.node.last_pose.pose.pose.position.y
                z_quat = self.node.last_pose.pose.pose.orientation.z
                w_quat = self.node.last_pose.pose.pose.orientation.w
                
                yaw = math.atan2(2.0 * (w_quat * z_quat), 1.0 - 2.0 * (z_quat * z_quat))
                yaw_deg = math.degrees(yaw)
                
                self.pos_var.set(f"X: {x:.3f}  |  Y: {y:.3f}  |  Yaw: {yaw_deg:.1f}°")
                
                # Show current click on canvas
                cx = self.canvas_width // 2 + x * self.scale
                cy = self.canvas_height // 2 - y * self.scale
                
                self.canvas.delete("current")
                self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, outline="#FF6B6B", width=3, tags="current")
            
            self.root.after(200, refresh)
        
        self.root.after(200, refresh)
    
    def run(self):
        """Run the UI"""
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = PoseSetterNode()
    
    # Create UI in main thread
    ui = PoseSetterUI(node)
    
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
