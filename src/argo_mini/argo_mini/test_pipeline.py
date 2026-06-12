#!/usr/bin/env python3

import urllib.request
import json
import time

# The address where your dashboard.py server is running
API_URL = "http://localhost:8080/api/command"

def send_table_click(waypoint_id):
    """Sends a mock POST request exactly like the dashboard website does."""
    payload = {
        "command": "navigate",
        "args": {
            "waypoint": waypoint_id,
            "mode": "shift"
        }
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode('utf-8')
    
    print(f"\n[TESTING] Simulating user click: Table {waypoint_id}...")
    
    try:
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print(f"[SUCCESS] Command sent for Table {waypoint_id}!")
            else:
                print(f"[FAILED] Server responded with status code: {response.status}")
    except Exception as e:
        print(f"[ERROR] Could not connect to dashboard server: {e}")

def run_automated_test():
    print("="*50)
    print("      STARTING INTEGRATION PIPELINE TEST         ")
    print("="*50)
    print("Make sure both dashboard.py and waypoint_manager.py are running!")
    
    # --- STEP 1: Click Table 1 ---
    send_table_click(1)
    print("Waiting 10 seconds for robot to drive towards Table 1...")
    time.sleep(10)
    
    # --- STEP 2: Click Table 2 ---
    send_table_click(2)
    print("Waiting 10 seconds for robot to alter course to Table 2...")
    time.sleep(10)
    
    # --- STEP 3: Return to Base Station ---
    send_table_click(0)
    print("\n[TEST FINISHED] Sent return to base command.")
    print("="*50)

if __name__ == "__main__":
    run_automated_test()