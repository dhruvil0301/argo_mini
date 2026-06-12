"""
Argo NLP Agent — LangChain + GPT-4o-mini WebSocket server.

Integrates with dashboard.py / admin_panel.py via:
  - WebSocket (port 8765): voice client + admin panel bridge connect here
  - HTTP POST to /api/argo/events: robot commands forwarded to ROS2 via dashboard
"""

import asyncio
import websockets
import json
import os
import time
import re
import urllib.request
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DASHBOARD_URL  = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:8080")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env file.")
    exit(1)

# ==========================================
# 1. STATE MACHINE
# ==========================================
class ArgoStateMachine:
    def __init__(self):
        self.current_waypoint   = "docking_station"
        self.status             = "BOOT_COMPLETE"
        self.task_queue         = []
        self.active_order       = []
        self.failed_stt_attempts = 0
        self.last_speech_time   = time.time()
        self.blocked_start_time = None
        self.session_active     = False
        self.is_activated       = False   # toggled by ADMIN_START / wake word
        self.person_tracking_active = False

    def get_time_of_day_greeting(self) -> str:
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "Good morning. Welcome to our restaurant. How may I assist you today?"
        elif 12 <= hour < 17:
            return "Good afternoon. Welcome. How may I assist you today?"
        else:
            return "Good evening. Welcome. How may I assist you tonight?"

    def add_to_queue(self, item_name: str, target_waypoint: str):
        task = {
            "task_id": int(time.time()),
            "item":    item_name,
            "target":  target_waypoint,
            "timestamp": time.time()
        }
        self.task_queue.append(task)
        print(f"[SYSTEM QUEUE] Appended Level 3 task: {task}")


state = ArgoStateMachine()
connected_clients: set = set()

# ==========================================
# 2. DASHBOARD HTTP BRIDGE
# ==========================================
def _post_to_dashboard_sync(payload: dict):
    """Blocking POST to the dashboard HTTP event endpoint."""
    url = f"{DASHBOARD_URL.rstrip('/')}/api/argo/events"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
    except Exception as exc:
        print(f"[AGENT] Dashboard notify failed: {exc}")


async def send_to_ui(payload: dict):
    """Non-blocking: runs HTTP POST in a thread so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _post_to_dashboard_sync, payload)

# ==========================================
# 3. TOOLS
# ==========================================
@tool
async def navigate_to(waypoint_name: str) -> str:
    """Dispatches Argo to a specific SLAM waypoint (e.g. 'table_1', 'table_2', 'kitchen', 'docking_station', 'restrooms_waypoint')."""
    state.status = "NAVIGATING"
    state.current_waypoint = waypoint_name
    await send_to_ui({
        "event": "ROBOT_COMMAND",
        "command": "NAVIGATE",
        "parameters": {"target_waypoint": waypoint_name}
    })
    return f"Autonomous navigation initiated to {waypoint_name}."


@tool
async def submit_order_to_database(items: list, table_id: str) -> str:
    """Submits the confirmed guest order list to the Kitchen Display System (KDS) database."""
    state.active_order = items
    await send_to_ui({
        "event": "KDS_UPDATE",
        "command": "SUBMIT_ORDER",
        "parameters": {
            "table_id": table_id,
            "items":    items,
            "status":   "pending"
        }
    })
    return f"Order payload successfully submitted to KDS database for Table {table_id}."


@tool
async def process_refill_queue(item_name: str, table_id: str) -> str:
    """Handles water, roti, and beverage refill requests based on system load (FSM Priority Queueing)."""
    if state.status in ["IDLE", "BOOT_COMPLETE"] and len(state.task_queue) == 0:
        state.status = "BUSY"
        await send_to_ui({
            "event": "ROBOT_COMMAND",
            "command": "NAVIGATE",
            "parameters": {"target_waypoint": "kitchen"}
        })
        return "FREE_STATUS_RESOLVED: Navigating directly to the service station to fetch your request."
    else:
        state.add_to_queue(item_name, f"table_{table_id}")
        await send_to_ui({
            "event": "KDS_UPDATE",
            "command": "ALERT_STAFF",
            "parameters": {
                "message":  f"Table {table_id} requested {item_name}. Argo is busy.",
                "priority": "medium"
            }
        })
        return "BUSY_STATUS_RESOLVED: Task appended to priority queue. Waitstaff alerted via KDS relay."


@tool
async def alert_floor_manager(table_id: str, reason: str) -> str:
    """Alerts the restaurant floor manager for priority human assistance."""
    await send_to_ui({
        "event":    "MANAGER_ALERT",
        "reason":   reason,
        "table_id": table_id,
    })
    return f"Floor manager alert dispatched for Table {table_id} due to: {reason}."


tools    = [navigate_to, submit_order_to_database, process_refill_queue, alert_floor_manager]
tool_map = {t.name: t for t in tools}

llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 4. SEMANTIC PRE-ROUTER
# ==========================================
def triaged_semantic_router(user_input: str, current_waypoint: str) -> tuple:
    clean_input = user_input.lower().strip()

    if clean_input in ["argo", "hey argo", "hey robot", "hello argo", "hi argo",
                       "good morning", "good afternoon", "good evening"]:
        return state.get_time_of_day_greeting(), None

    if re.search(r"\b(how are you|who created you|about yourself)\b", clean_input):
        return (
            "I am operating at full capacity, thank you for asking. "
            "I hope you are having an excellent dining experience. "
            "How may I assist with your service?",
            None
        )

    if re.search(r"\b(restroom|restrooms|bathroom|washroom|toilet|handwash)\b", clean_input):
        return (
            "Our restrooms are located down the main hallway, past the kitchen on your left. "
            "Would you like me to guide you there?",
            {"type": "suggest_navigation", "target": "restrooms_waypoint"}
        )

    if state.session_active and current_waypoint != "restrooms_waypoint":
        if clean_input in ["yes please", "yes", "guide me", "show me", "help me"]:
            return "Certainly. Please follow me.", {"type": "execute_navigation", "target": "restrooms_waypoint"}
        elif clean_input in ["no thank you", "no thanks", "no", "i can find it"]:
            return "Very well. Enjoy your evening.", {"type": "deactivate_session"}

    return None, None

# ==========================================
# 5. CONVERSATIONAL ENGINE
# ==========================================
system_prompt = """
You are Argo, the highly professional, polished robotic waiter.
You are currently located at: {current_waypoint}.
Your active task queue currently contains: {queue_len} pending tasks.

You must strictly adhere to the following dialogue guidelines:

1. GREETINGS:
   - Always welcome guests with: "{greeting_string}"

2. ORDERING:
   - For orders: "Certainly. I have recorded [Menu Items] for Table [X]. Shall I submit this order directly to the kitchen?"
   - Upon confirmation: "Excellent. Your order has been submitted to the chef. I will return once it is prepared." and invoke 'submit_order_to_database'.
   - If user updates order mid-way: "Understood. I have updated your order to two [Menu Items] for Table [X]. Shall I submit this to the kitchen now?"

3. OUT OF STOCK (Apologize Sincerely - NO Substitutions):
   - If an item is out of stock: "My apologies, but our [Item] is currently unavailable today. I have removed the item from your order. Shall I submit the rest of your selections to the kitchen?"
   - Never suggest alternative dishes or items.

4. KITCHEN ARRIVAL:
   - Upon reaching the kitchen waypoint: "Good day, Chef. I have submitted the order for Table [X]. Please let me know once it is loaded on my trays."
   - If Chef reports mid-meal stock out: "Understood, Chef. I will return to Table [X] immediately to inform the guests."

5. REFILLS & QUEUEING (WATER/ROTI):
   - When asked for a refill, execute the 'process_refill_queue' tool immediately.
   - If FREE_STATUS_RESOLVED: "Certainly. I will fetch a [Water Bottle / Extra Roti] for Table [X] immediately."
   - If BUSY_STATUS_RESOLVED: "Certainly. I have queued a bottle of water for Table [X]. I must assist [Next Table] first, but I will return with your water immediately afterward. I have also notified our service staff in case they can reach you sooner."

6. STAFF CALLS:
   - Manager Request: "I understand. I have sent an immediate, high-priority alert to our floor manager. They will be here at Table [X] to assist you momentarily." and invoke 'alert_floor_manager'.

7. TIMEOUT / SILENCE:
   - If [TIMEOUT_NO_RESPONSE]: politely remind the guest you are still present and ready to assist. Keep it brief and natural.
"""


async def run_conversational_engine(user_input: str, chat_history: list) -> tuple:
    active_prompt = system_prompt.format(
        current_waypoint=state.current_waypoint,
        queue_len=len(state.task_queue),
        greeting_string=state.get_time_of_day_greeting()
    )

    if user_input == "[TIMEOUT_NO_RESPONSE]":
        human_msg = HumanMessage(content="[Silence - no response from customer]")
        messages = [SystemMessage(content=active_prompt)] + chat_history + [
            human_msg,
            SystemMessage(content="The customer did not respond. Politely remind them you are still here and ask if they need anything. Keep it short and natural.")
        ]
    else:
        human_msg = HumanMessage(content=user_input)
        messages = [SystemMessage(content=active_prompt)] + chat_history + [human_msg]

    response = await llm_with_tools.ainvoke(messages)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name in tool_map:
                print(f"[ACTION] Triggering Tool: {tool_name} with args: {tool_args}")
                tool_output = await tool_map[tool_name].ainvoke(tool_args)
                messages.append(response)
                messages.append(AIMessage(content=f"System confirmation of tool run: {tool_output}"))
                response = await llm_with_tools.ainvoke(messages)

    return response.content, chat_history + [human_msg, AIMessage(content=response.content)]

# ==========================================
# 6. BACKGROUND HEARTBEAT
# ==========================================
async def background_monitoring_heartbeat():
    while True:
        await asyncio.sleep(1.0)
        if state.status == "BLOCKED" and state.blocked_start_time:
            elapsed = time.time() - state.blocked_start_time
            if elapsed >= 25.0:
                print("\n[PATH BLOCKED - 25 s] Alerting staff via dashboard...")
                await send_to_ui({
                    "event": "KDS_UPDATE",
                    "command": "ALERT_STAFF",
                    "parameters": {
                        "table_id": state.current_waypoint[-1] if state.current_waypoint[-1].isdigit() else "?",
                        "alert":    "navigation_path_blocked",
                        "priority": "high"
                    }
                })
                state.status = "IDLE"
                state.blocked_start_time = None

# ==========================================
# 7. WEBSOCKET SERVER
# ==========================================
async def handle_connection(websocket):
    connected_clients.add(websocket)
    print("\n[SERVER] New client connected!")

    state.status = "IDLE"
    state.last_speech_time = time.time()
    chat_history = []

    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                data = {"text": raw_message}

            control_event = data.get("event", "")

            # ── Admin panel control events ──────────────────────────────
            if control_event == "ADMIN_START":
                state.is_activated  = True
                state.session_active = True
                print("[SERVER] ADMIN_START — session activated by admin panel.")
                for client in list(connected_clients):
                    if client != websocket:
                        try:
                            await client.send(json.dumps({"event": "ADMIN_WAKE"}))
                        except Exception:
                            pass
                continue

            if control_event == "ADMIN_STOP":
                state.is_activated   = False
                state.session_active = False
                print("[SERVER] ADMIN_STOP — session deactivated.")
                for client in list(connected_clients):
                    if client != websocket:
                        try:
                            await client.send(json.dumps({"event": "ADMIN_SLEEP"}))
                        except Exception:
                            pass
                continue

            if control_event == "SIMULATE_BLOCKAGE":
                state.status = "BLOCKED"
                state.blocked_start_time = time.time()
                continue

            if control_event == "SIMULATE_CLEAR":
                state.status = "IDLE"
                state.blocked_start_time = None
                continue

            # ── Speech / text input ─────────────────────────────────────
            user_input = data.get("text", "").strip()
            if not user_input:
                continue

            state.last_speech_time = time.time()

            # Log user turn to dashboard conversation log
            await send_to_ui({
                "event":  "USER_INPUT",
                "text":   user_input,
                "source": data.get("source", "voice")
            })

            if user_input == "":
                state.failed_stt_attempts += 1
                if state.failed_stt_attempts == 1:
                    err_txt = "My apologies, the room is quite lively and I didn't quite catch that. Could you please repeat your request?"
                elif state.failed_stt_attempts >= 2:
                    err_txt = (f"I am still having trouble hearing your request over the background noise. "
                               f"I have notified our service staff to assist you directly at "
                               f"Table {state.current_waypoint[-1] if state.current_waypoint[-1].isdigit() else '[X]'}. "
                               f"Thank you for your patience.")
                    await send_to_ui({"event": "KDS_UPDATE", "command": "ALERT_STAFF",
                                       "parameters": {"table_id": state.current_waypoint[-1],
                                                      "alert": "noise_interference_assistance", "priority": "medium"}})
                    state.failed_stt_attempts = 0
                else:
                    continue
                await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": err_txt}))
                await send_to_ui({"event": "SPEECH_RESPONSE", "text": err_txt})
                continue

            state.failed_stt_attempts = 0
            if user_input == "[TIMEOUT_NO_RESPONSE]":
                print("[VOICE INPUT] Silence timeout from customer.")
            else:
                print(f"[VOICE INPUT] Received: '{user_input}'")

            # Semantic pre-router (fast path — bypasses LLM)
            direct_reply, action_directive = triaged_semantic_router(user_input, state.current_waypoint)

            if direct_reply:
                print(f"[ROUTER] Direct match: {direct_reply}")
                await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": direct_reply}))
                await send_to_ui({"event": "SPEECH_RESPONSE", "text": direct_reply})
                if action_directive:
                    if action_directive["type"] == "execute_navigation":
                        await navigate_to.ainvoke({"waypoint_name": action_directive["target"]})
                    elif action_directive["type"] == "deactivate_session":
                        state.session_active = False
                continue

            # LLM agent (slow path)
            print("[AGENT] Querying model over network...")
            response_text, chat_history = await run_conversational_engine(user_input, chat_history)
            print(f"Argo says: {response_text}")
            await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": response_text}))
            await send_to_ui({"event": "SPEECH_RESPONSE", "text": response_text})

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

    except websockets.ConnectionClosed:
        print("[SERVER] Client disconnected.")
    finally:
        connected_clients.discard(websocket)


async def main():
    print("[SERVER] Starting Argo Gateway on ws://127.0.0.1:8765 ...")
    asyncio.create_task(background_monitoring_heartbeat())
    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
