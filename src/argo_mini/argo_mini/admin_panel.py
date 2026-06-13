"""
Admin Panel service — menu management, table orders, NLP session handling.

All restaurant/admin business logic lives here. dashboard.py forwards
/api/restaurant* and /api/argo/* requests to this module as a thin gateway.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import re

from dotenv import dotenv_values, load_dotenv

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, '.env')
DATA_FILE = os.path.join(PROJECT_DIR, 'restaurant_data.json')
AGENT_SCRIPT = os.path.join(PROJECT_DIR, 'rest_agent.py')
VOICE_SCRIPT = os.path.join(PROJECT_DIR, 'voice_client.py')
NAV_SCRIPT_PATH = os.environ.get('ARGO_NAV_SCRIPT_PATH', '/home/argo/argo_mini_ws/start_argo_nav.py')
AGENT_WS_URL = os.environ.get('ARGO_AGENT_WS_URL', 'ws://127.0.0.1:8765')
MAX_LOG_ENTRIES = 200
AGENT_STARTUP_TIMEOUT_SEC = 15

# Load .env for this process and for child agent subprocesses.
load_dotenv(ENV_FILE)

# Routes owned by this module (gateway forwards these paths here)
ADMIN_GET_ROUTES = {'/api/restaurant', '/api/argo/nlp'}
ADMIN_POST_ROUTES = {
    '/api/restaurant/menu',
    '/api/restaurant/order',
    '/api/restaurant/order/action',
    '/api/restaurant/table/reset',
    '/api/argo/nav/start',
    '/api/argo/nav/stop',
    '/api/argo/nlp/start',
    '/api/argo/nlp/stop',
    '/api/argo/nlp/wakeup',
    '/api/argo/nlp/message',
    '/api/argo/events',
}


class _ExternalProcess:
    """Sentinel: makes get_state() report voiceClientRunning=True
    when the voice client was started outside the admin panel."""
    def poll(self): return None   # None = still running


def _strip_ansi(text):
    return ANSI_RE.sub("", text or "")


class RestaurantManager:
    """Persistent store for menu items and per-table order sessions."""

    def __init__(self, data_file=DATA_FILE):
        self.data_file = data_file
        self._lock = threading.Lock()
        self.data = self._load()

    def _default_data(self):
        return {
            "menu": [
                {"id": "1", "name": "Margherita Pizza", "price": 12.99,
                 "description": "Classic tomato, mozzarella & basil", "available": True},
                {"id": "2", "name": "Caesar Salad", "price": 8.50,
                 "description": "Romaine, parmesan, croutons & dressing", "available": True},
                {"id": "3", "name": "Grilled Chicken", "price": 15.99,
                 "description": "Herb-marinated chicken with seasonal vegetables", "available": True},
                {"id": "4", "name": "Chocolate Lava Cake", "price": 6.99,
                 "description": "Warm molten chocolate dessert", "available": True},
            ],
            "tables": {}
        }

    def _load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if "menu" not in data:
                        data["menu"] = self._default_data()["menu"]
                    if "tables" not in data:
                        data["tables"] = {}
                    return data
            except Exception as e:
                print(f"[AdminPanel] Failed to load data: {e}")
        return self._default_data()

    def _save(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[AdminPanel] Failed to save data: {e}")

    def _ensure_table(self, table_id):
        key = str(table_id)
        if key not in self.data["tables"]:
            self.data["tables"][key] = {"sessionId": 1, "orders": []}
        return self.data["tables"][key]

    def get_state(self):
        with self._lock:
            return json.loads(json.dumps(self.data))

    def add_menu_item(self, name, price, description):
        with self._lock:
            item = {
                "id": str(uuid.uuid4())[:8],
                "name": str(name).strip(),
                "price": round(float(price), 2),
                "description": str(description).strip(),
                "available": True
            }
            self.data["menu"].append(item)
            self._save()
            return item

    def update_menu_item(self, item_id, name=None, price=None, description=None):
        with self._lock:
            for item in self.data["menu"]:
                if item["id"] == item_id:
                    if name is not None:
                        item["name"] = str(name).strip()
                    if price is not None:
                        item["price"] = round(float(price), 2)
                    if description is not None:
                        item["description"] = str(description).strip()
                    self._save()
                    return item
            return None

    def delete_menu_item(self, item_id):
        with self._lock:
            before = len(self.data["menu"])
            self.data["menu"] = [m for m in self.data["menu"] if m["id"] != item_id]
            if len(self.data["menu"]) < before:
                self._save()
                return True
            return False

    def toggle_menu_availability(self, item_id):
        with self._lock:
            for item in self.data["menu"]:
                if item["id"] == item_id:
                    item["available"] = not item.get("available", True)
                    self._save()
                    return item
            return None

    def place_order(self, table_id, items):
        with self._lock:
            table = self._ensure_table(table_id)
            order_items = []
            total = 0.0
            menu_by_id = {m["id"]: m for m in self.data["menu"]}

            for entry in items:
                menu_id = str(entry.get("menuItemId", ""))
                qty = int(entry.get("qty", 1))
                if qty < 1:
                    continue
                menu_item = menu_by_id.get(menu_id)
                if not menu_item or not menu_item.get("available", True):
                    continue
                line_total = menu_item["price"] * qty
                total += line_total
                order_items.append({
                    "menuItemId": menu_id,
                    "name": menu_item["name"],
                    "price": menu_item["price"],
                    "qty": qty,
                    "lineTotal": round(line_total, 2)
                })

            if not order_items:
                return None

            order = {
                "id": str(uuid.uuid4())[:8],
                "items": order_items,
                "status": "pending",
                "total": round(total, 2),
                "createdAt": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            table["orders"].append(order)
            self._save()
            return order

    def order_action(self, table_id, order_id, action):
        with self._lock:
            table = self._ensure_table(table_id)
            for order in table["orders"]:
                if order["id"] == order_id and order["status"] == "pending":
                    if action == "confirm":
                        order["status"] = "confirmed"
                    elif action == "cancel":
                        order["status"] = "cancelled"
                    else:
                        return None
                    self._save()
                    return order
            return None

    def reset_table(self, table_id):
        with self._lock:
            table = self._ensure_table(table_id)
            table["sessionId"] = table.get("sessionId", 1) + 1
            table["orders"] = []
            self._save()
            return table


class ArgoNlpManager:
  """Starts/stops the NLP agent process and bridges admin chat to its WebSocket."""

  def __init__(self):
    self._lock = threading.Lock()
    self._process = None
    self._voice_process = None
    self._voice_log_thread = None
    self._process_log_thread = None
    self._nav_log_thread = None
    self._bridge_thread = None
    self._bridge_loop = None
    self._bridge_ws = None
    self._bridge_ready = threading.Event()
    self._response_waiters = []
    self._nav_process = None
    self._nav_output = []
    self.nav_status_message = "Navigation launcher is idle."
    self.conversation_log = []
    self.manager_alerts = []
    self.session_active = False
    self.agent_connected = False
    self.robot_status = "IDLE"
    self.current_waypoint = "docking_station"
    self.started_at = None
    self.last_error = None
    self.voice_state = "offline"
    self.argo_awake = False
    self._process_output = []
    self._voice_output = []

  def _build_agent_env(self):
    env = os.environ.copy()
    if os.path.exists(ENV_FILE):
      env.update({k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None})
    env.setdefault("DASHBOARD_URL", "http://127.0.0.1:8080")
    return env

  def _missing_api_key_message(self):
    return (
      "OPENAI_API_KEY is not set. Create a .env file in the argo project folder with:\n"
      "OPENAI_API_KEY=sk-your-key-here"
    )

  def _record_process_output(self, line):
    clean = (line or "").strip()
    if not clean:
      return
    with self._lock:
      self._process_output.append(clean)
      self._process_output = self._process_output[-30:]
      if "ERROR:" in clean or "Traceback" in clean:
        self.last_error = clean

  def _start_process_log_reader(self):
    if not self._process or not self._process.stdout:
      return

    def _reader():
      try:
        for line in self._process.stdout:
          self._record_process_output(line)
          print(f"[ArgoAgent] {line.rstrip()}")
      except Exception as exc:
        print(f"[ArgoAgent] Log reader stopped: {exc}")

    self._process_log_thread = threading.Thread(target=_reader, daemon=True)
    self._process_log_thread.start()

  def _process_failure_message(self):
    with self._lock:
      if self.last_error:
        return self.last_error
      tail = "\n".join(self._process_output[-5:])
      if tail:
        return f"Agent process exited unexpectedly.\n{tail}"
    return "Agent process exited before the WebSocket server started."

  def get_state(self):
    with self._lock:
      running = self._process is not None and self._process.poll() is None
      return {
        "running": running,
        "sessionActive": self.session_active,
        "agentConnected": self.agent_connected,
        "robotStatus": self.robot_status,
        "currentWaypoint": self.current_waypoint,
        "startedAt": self.started_at,
        "lastError": self.last_error,
        "voiceState": self.voice_state,
        "voiceClientRunning": (
          self._voice_process is not None and self._voice_process.poll() is None
        ),
        "navRunning": (
          self._nav_process is not None and self._nav_process.poll() is None
        ),
        "navStatusMessage": self.nav_status_message,
        "navLogs": list(self._nav_output),
        "argoAwake": self.argo_awake,
        "conversationLog": list(self.conversation_log),
        "managerAlerts": list(self.manager_alerts),
      }

  def _append_log(self, role, text, source="robot"):
    if not text:
      return
    entry = {
      "id": str(uuid.uuid4())[:8],
      "role": role,
      "text": text,
      "source": source,
      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    self.conversation_log.append(entry)
    if len(self.conversation_log) > MAX_LOG_ENTRIES:
      self.conversation_log = self.conversation_log[-MAX_LOG_ENTRIES:]

  def handle_agent_event(self, payload):
    with self._lock:
      event = payload.get("event")
      if event == "SPEECH_RESPONSE":
        self._append_log("argo", payload.get("text", ""), source="robot")
      elif event == "USER_INPUT":
        self._append_log("guest", payload.get("text", ""), source=payload.get("source", "voice"))
      elif event == "MANAGER_ALERT":
        alert = {
          "id": str(uuid.uuid4())[:8],
          "reason": payload.get("reason", "UNKNOWN"),
          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
          "acknowledged": False,
        }
        self.manager_alerts.insert(0, alert)
        self.manager_alerts = self.manager_alerts[:50]
      elif event == "STATUS_UPDATE":
        self.robot_status = payload.get("status", self.robot_status)
        self.current_waypoint = payload.get("waypoint", self.current_waypoint)
        self.agent_connected = payload.get("connected", self.agent_connected)
      elif event == "VOICE_STATE":
        new_state = payload.get("state", self.voice_state)
        self.voice_state = new_state
        # If the voice client was launched externally (not via admin panel),
        # use a sentinel so voiceClientRunning=True and UI exits 'offline'.
        if self._voice_process is None and new_state != "offline":
          self._voice_process = _ExternalProcess()
        elif new_state == "offline" and isinstance(self._voice_process, _ExternalProcess):
          self._voice_process = None
      return {"status": "ok"}

  def start_session(self):
    with self._lock:
      if self._process and self._process.poll() is None:
        self.session_active = True
        self.last_error = None
        self._send_control_async("ADMIN_START")
        env = self._build_agent_env()
        self._start_voice_client(env)
        return {"status": "ok", "message": "Voice session resumed on running agent."}

      # Detect an externally-launched agent already listening on the port.
      # If found, adopt it without spawning a new process (avoids port conflict).
      import socket as _socket
      _s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
      _s.settimeout(1)
      _port_open = _s.connect_ex(('127.0.0.1', 8765)) == 0
      _s.close()

      if _port_open:
        print("[ArgoNlp] External agent detected on port 8765 — adopting without spawn.")
        self._process = _ExternalProcess()
        self.session_active = True
        self.last_error = None
        self._start_bridge_thread()
      else:
        if not os.path.exists(AGENT_SCRIPT):
          return {"status": "error", "message": f"Agent script not found: {AGENT_SCRIPT}"}

        env = self._build_agent_env()
        if not env.get("OPENAI_API_KEY"):
          self.last_error = self._missing_api_key_message()
          self.session_active = False
          return {"status": "error", "message": self.last_error}

        self.last_error = None
        self._process_output = []
        self._process = subprocess.Popen(
          [sys.executable, AGENT_SCRIPT],
          cwd=PROJECT_DIR,
          env=env,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,
        )
        self._start_process_log_reader()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._start_bridge_thread()
        self.session_active = True

    if not self._wait_for_bridge(AGENT_STARTUP_TIMEOUT_SEC):
      with self._lock:
        self.session_active = False
        if self._process and not isinstance(self._process, _ExternalProcess) \
                and self._process.poll() is None:
          self._process.terminate()
        elif self._process and not isinstance(self._process, _ExternalProcess) \
                and self._process.poll() is not None:
          self.last_error = self._process_failure_message()
        self._process = None
        self._stop_bridge_locked()
        message = self.last_error or (
          f"Could not connect to agent at {AGENT_WS_URL} within {AGENT_STARTUP_TIMEOUT_SEC}s."
        )
      return {"status": "error", "message": message}

    with self._lock:
      env = self._build_agent_env()
    voice_msg = self._start_voice_client(env)
    message = "Argo voice communication started. Say \"Hey Argo\" to begin."
    if voice_msg:
      message = f"{message} Warning: {voice_msg}"
    return {"status": "ok", "message": message}

  def _start_voice_client(self, env):
    if not os.path.exists(VOICE_SCRIPT):
      return "Voice client script not found."

    with self._lock:
      if self._voice_process and self._voice_process.poll() is None:
        return None

      self._voice_output = []
      self._voice_process = subprocess.Popen(
        [sys.executable, VOICE_SCRIPT],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      self._start_voice_log_reader()

    time.sleep(1.5)
    with self._lock:
      if self._voice_process and self._voice_process.poll() is not None:
        tail = "\n".join(self._voice_output[-5:])
        self.voice_state = "offline"
        return tail or "Voice client exited immediately. Install requirements-voice.txt and check microphone."
    return None

  def _start_voice_log_reader(self):
    if not self._voice_process or not self._voice_process.stdout:
      return

    def _reader():
      try:
        for line in self._voice_process.stdout:
          clean = line.rstrip()
          if clean:
            with self._lock:
              self._voice_output.append(clean)
              self._voice_output = self._voice_output[-30:]
            print(f"[ArgoVoice] {clean}")
      except Exception as exc:
        print(f"[ArgoVoice] Log reader stopped: {exc}")

    self._voice_log_thread = threading.Thread(target=_reader, daemon=True)
    self._voice_log_thread.start()

  def _start_nav_log_reader(self):
    if not self._nav_process or not self._nav_process.stdout:
      return

    def _reader():
      try:
        for line in self._nav_process.stdout:
          clean = line.rstrip()
          if clean:
            plain = _strip_ansi(clean)
            with self._lock:
              self._nav_output.append(plain)
              self._nav_output = self._nav_output[-120:]
              if "Navigation stack is LIVE" in plain or "All Systems Nominal" in plain:
                self.nav_status_message = "Navigation ready."
              elif "Starting" in plain:
                self.nav_status_message = plain
              elif "Timeout" in plain or "abort" in plain.lower():
                self.nav_status_message = plain
            print(f"[ArgoNav] {plain}")
      except Exception as exc:
        print(f"[ArgoNav] Log reader stopped: {exc}")

    self._nav_log_thread = threading.Thread(target=_reader, daemon=True)
    self._nav_log_thread.start()

  def _stop_voice_client_locked(self):
    if self._voice_process and self._voice_process.poll() is None:
      self._voice_process.terminate()
      try:
        self._voice_process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self._voice_process.kill()
      self._voice_process = None
      self.voice_state = "offline"

  def start_navigation(self):
    with self._lock:
      if self._nav_process and self._nav_process.poll() is None:
        self.nav_status_message = "Navigation launcher is already running."
        return {"status": "ok", "message": "Navigation script is already running."}

    if not os.path.exists(NAV_SCRIPT_PATH):
      with self._lock:
        self.nav_status_message = f"Navigation script not found: {NAV_SCRIPT_PATH}"
      return {"status": "error", "message": f"Navigation script not found: {NAV_SCRIPT_PATH}"}

    nav_dir = os.path.dirname(NAV_SCRIPT_PATH) or PROJECT_DIR
    nav_cmd = NAV_SCRIPT_PATH
    if nav_cmd.endswith(".py"):
      cmd = [sys.executable, nav_cmd]
    elif os.access(NAV_SCRIPT_PATH, os.X_OK):
      cmd = [nav_cmd]
    else:
      cmd = ['bash', nav_cmd]

    try:
      with self._lock:
        self._nav_output = []
        self.nav_status_message = "Launching navigation stack..."
      self._nav_process = subprocess.Popen(
        cmd,
        cwd=nav_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      self._start_nav_log_reader()
    except Exception as exc:
      self._nav_process = None
      with self._lock:
        self.nav_status_message = f"Navigation launch failed: {exc}"
      return {"status": "error", "message": str(exc)}

    time.sleep(0.5)
    if self._nav_process.poll() is not None:
      with self._lock:
        self.nav_status_message = f"Navigation script exited immediately: {NAV_SCRIPT_PATH}"
      return {
        "status": "error",
        "message": f"Navigation script exited immediately: {NAV_SCRIPT_PATH}",
      }

    return {
      "status": "ok",
      "message": f"Started navigation script: {NAV_SCRIPT_PATH}",
    }

  def stop_navigation(self):
    with self._lock:
      if not self._nav_process or self._nav_process.poll() is not None:
        self._nav_process = None
        self.nav_status_message = "Navigation launcher is already stopped."
        return {"status": "ok", "message": "Navigation script is already stopped."}

      self._nav_process.terminate()
      try:
        self._nav_process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self._nav_process.kill()
      self._nav_process = None
      self.nav_status_message = "Navigation launcher stopped."
      return {"status": "ok", "message": "Stopped navigation script."}

  def _wait_for_bridge(self, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
      if self._process and self._process.poll() is not None:
        return False
      if self._bridge_ready.wait(timeout=0.5):
        return True
    return False

  def _stop_bridge_locked(self):
    loop = self._bridge_loop
    if loop and loop.is_running():
      asyncio.run_coroutine_threadsafe(self._close_bridge(), loop)
    self._bridge_ready.clear()

  def stop_session(self):
    with self._lock:
      self.session_active = False
      self._send_control_async("ADMIN_STOP")
      self._stop_bridge_locked()
      if self._process and not isinstance(self._process, _ExternalProcess) \
              and self._process.poll() is None:
        self._process.terminate()
        try:
          self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          self._process.kill()
      self._stop_voice_client_locked()
      self._process = None
      self.agent_connected = False
      self.argo_awake = False
      return {"status": "ok", "message": "Argo voice session stopped."}

  def toggle_wake(self):
    with self._lock:
      running = self._process is not None and self._process.poll() is None
      if not running:
        return {"status": "error", "message": "Agent is not running. Start communication first."}
      if not self.argo_awake:
        self.argo_awake = True
        self._send_control_async("ADMIN_START")
        return {"status": "ok", "argoAwake": True, "message": "Argo is now awake and listening."}
      else:
        self.argo_awake = False
        self._send_control_async("ADMIN_STOP")
        return {"status": "ok", "argoAwake": False, "message": "Argo has returned to sleep mode."}

  def send_message(self, text):
    clean = str(text or "").strip()
    if not clean:
      return {"status": "error", "message": "Message text is required."}

    with self._lock:
      if not self.session_active:
        return {"status": "error", "message": "Start Argo communication before sending messages."}
      if not self._bridge_ready.wait(timeout=8):
        return {"status": "error", "message": "Agent bridge is not connected yet."}

      self._append_log("guest", clean, source="admin")
      waiter = {"event": threading.Event(), "text": None}
      self._response_waiters.append(waiter)

    try:
      self._send_text_async(clean)
      if not waiter["event"].wait(timeout=30):
        return {"status": "error", "message": "Timed out waiting for Argo response."}
      return {"status": "ok", "response": waiter["text"]}
    except Exception as e:
      return {"status": "error", "message": str(e)}
    finally:
      with self._lock:
        if waiter in self._response_waiters:
          self._response_waiters.remove(waiter)

  def _start_bridge_thread(self):
    self._bridge_ready.clear()
    self._bridge_thread = threading.Thread(target=self._bridge_worker, daemon=True)
    self._bridge_thread.start()

  def _bridge_worker(self):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    self._bridge_loop = loop
    loop.run_until_complete(self._bridge_main())

  async def _bridge_main(self):
    import websockets

    while self.session_active:
      if self._process and self._process.poll() is not None:
        with self._lock:
          self.last_error = self._process_failure_message()
          self.session_active = False
        print(f"[ArgoNlp] Agent process exited: {self.last_error}")
        return

      try:
        async with websockets.connect(AGENT_WS_URL, open_timeout=3) as ws:
          self._bridge_ws = ws
          self._bridge_ready.set()
          print("[ArgoNlp] Bridge connected to agent.")
          await ws.send(json.dumps({"event": "ADMIN_START"}))
          async for raw in ws:
            try:
              data = json.loads(raw)
            except json.JSONDecodeError:
              continue
            if data.get("event") == "SPEECH_RESPONSE":
              text = data.get("text", "")
              with self._lock:
                self._append_log("argo", text, source="robot")
                for waiter in list(self._response_waiters):
                  waiter["text"] = text
                  waiter["event"].set()
      except Exception as e:
        if not self.session_active:
          return
        if self._process and self._process.poll() is not None:
          with self._lock:
            self.last_error = self._process_failure_message()
            self.session_active = False
          return
        self._bridge_ready.clear()
        await asyncio.sleep(1.0)
        if self._bridge_ready.is_set():
          continue
        print(f"[ArgoNlp] Waiting for agent at {AGENT_WS_URL}...")

    print("[ArgoNlp] Bridge stopped.")

  async def _close_bridge(self):
    if self._bridge_ws:
      await self._bridge_ws.close()
    self._bridge_ws = None

  def _send_control_async(self, event_name):
    if self._bridge_loop and self._bridge_loop.is_running() and self._bridge_ws:
      asyncio.run_coroutine_threadsafe(
        self._bridge_ws.send(json.dumps({"event": event_name})),
        self._bridge_loop,
      )

  def _send_text_async(self, text):
    if self._bridge_loop and self._bridge_loop.is_running() and self._bridge_ws:
      asyncio.run_coroutine_threadsafe(
        self._bridge_ws.send(json.dumps({"text": text, "source": "admin"})),
        self._bridge_loop,
      )


class AdminPanelRouter:
    """HTTP route handler for all /api/restaurant* and /api/argo/* endpoints."""

    def __init__(self, manager=None, nlp_manager=None):
        self.manager = manager or RestaurantManager()
        self.nlp = nlp_manager or ArgoNlpManager()

    def is_admin_route(self, method, path):
        if method == 'GET':
            return path in ADMIN_GET_ROUTES
        if method == 'POST':
            return path in ADMIN_POST_ROUTES
        return False

    def handle_get(self, path):
        if path == '/api/restaurant':
            return 200, self.manager.get_state()
        if path == '/api/argo/nlp':
            return 200, self.nlp.get_state()
        return None

    def handle_post(self, path, post_data):
        try:
            req = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return 400, {"status": "error", "message": str(e)}

        if path == '/api/restaurant/menu':
            return self._handle_menu(req)
        if path == '/api/restaurant/order':
            return self._handle_place_order(req)
        if path == '/api/restaurant/order/action':
            return self._handle_order_action(req)
        if path == '/api/restaurant/table/reset':
            return self._handle_table_reset(req)
        if path == '/api/argo/nlp/start':
            return 200, self.nlp.start_session()
        if path == '/api/argo/nav/start':
            return 200, self.nlp.start_navigation()
        if path == '/api/argo/nav/stop':
            return 200, self.nlp.stop_navigation()
        if path == '/api/argo/nlp/stop':
            return 200, self.nlp.stop_session()
        if path == '/api/argo/nlp/wakeup':
            return 200, self.nlp.toggle_wake()
        if path == '/api/argo/nlp/message':
            return 200, self.nlp.send_message(req.get('text', ''))
        if path == '/api/argo/events':
            return 200, self.nlp.handle_agent_event(req)
        return None

    def _handle_menu(self, req):
        try:
            action = req.get('action')
            result = None

            if action == 'add':
                result = self.manager.add_menu_item(
                    req.get('name', ''), req.get('price', 0), req.get('description', ''))
            elif action == 'update':
                result = self.manager.update_menu_item(
                    req.get('id'), req.get('name'), req.get('price'), req.get('description'))
            elif action == 'delete':
                result = self.manager.delete_menu_item(req.get('id'))
            elif action == 'toggle':
                result = self.manager.toggle_menu_availability(req.get('id'))
            else:
                return 400, {"status": "error", "message": f"Unknown action: {action}"}

            return 200, {"status": "ok", "result": result}
        except Exception as e:
            print(f"[AdminPanel] Menu error: {e}")
            return 400, {"status": "error", "message": str(e)}

    def _handle_place_order(self, req):
        try:
            order = self.manager.place_order(req.get('table'), req.get('items', []))
            if order:
                return 200, {"status": "ok", "order": order}
            return 400, {"status": "error", "message": "No valid items"}
        except Exception as e:
            print(f"[AdminPanel] Order error: {e}")
            return 400, {"status": "error", "message": str(e)}

    def _handle_order_action(self, req):
        try:
            order = self.manager.order_action(
                req.get('table'), req.get('orderId'), req.get('action'))
            if order:
                return 200, {"status": "ok", "order": order}
            return 404, {"status": "error", "message": "Order not found"}
        except Exception as e:
            print(f"[AdminPanel] Order action error: {e}")
            return 400, {"status": "error", "message": str(e)}

    def _handle_table_reset(self, req):
        try:
            table = self.manager.reset_table(req.get('table'))
            return 200, {"status": "ok", "table": table}
        except Exception as e:
            print(f"[AdminPanel] Table reset error: {e}")
            return 400, {"status": "error", "message": str(e)}


# Singleton used by dashboard.py gateway
admin_router = AdminPanelRouter()


def handle_admin_get(path):
    """Gateway entry: returns (status_code, body_dict) or None."""
    return admin_router.handle_get(path)


def handle_admin_post(path, post_data):
    """Gateway entry: returns (status_code, body_dict) or None."""
    return admin_router.handle_post(path, post_data)


def is_admin_route(method, path):
    """Gateway entry: check if path belongs to admin panel."""
    return admin_router.is_admin_route(method, path)
