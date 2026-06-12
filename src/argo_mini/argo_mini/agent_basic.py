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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Load environment configurations from the project folder (same dir as this script).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:8080")


def _require_api_key():
    if OPENAI_API_KEY:
        return
    print(
        "ERROR: OPENAI_API_KEY not found. "
        f"Add it to {_SCRIPT_DIR}/.env or export it before starting the agent."
    )
    raise SystemExit(1)

# ==========================================
# 1. STATE MACHINE WITH REFINED SESSION FLOW
# ==========================================

class ArgoStateMachine:
    def __init__(self):
        self.current_waypoint = "docking_station"
        self.status = "IDLE"
        self.is_activated = False  # True when wake-word is triggered
        self.last_speech_time = time.time()
        self.failed_stt_attempts = 0  # Track unclear/empty inputs
        self.silence_prompt_count = 0  # Track silence prompts to prevent looping
        self.task_queue = []           # Priority queue for refill/delivery tasks
        self.active_order = []         # Current active food order items
        self.blocked_start_time = None # Physical navigation blockage timer

    def get_time_of_day_greeting(self) -> str:
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "Good morning. I am Argo. How may I assist you today?"
        elif 12 <= hour < 17:
            return "Good afternoon. I am Argo. How may I assist you today?"
        else:
            return "Good evening. I am Argo. How may I assist you tonight?"

    def add_to_queue(self, item_name: str, target_waypoint: str):
        task = {
            "task_id": int(time.time()),
            "item": item_name,
            "target": target_waypoint,
            "timestamp": time.time(),
        }
        self.task_queue.append(task)
        print(f"[SYSTEM QUEUE] Appended task: {task}")

state = ArgoStateMachine()
connected_clients = set()

# ==========================================
# 2. DIGITAL OUTBOUND PAYLOADS (DASHBOARD STUB)
# ==========================================

async def send_to_dashboard(payload: dict):
    """Forward agent events to dashboard.py gateway (admin panel + navigation)."""
    url = f"{DASHBOARD_URL.rstrip('/')}/api/argo/events"

    def _post():
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()

    try:
        await asyncio.get_running_loop().run_in_executor(None, _post)
    except Exception as exc:
        print(f"[DASHBOARD] Failed to deliver payload: {exc}")

# ==========================================
# 3. CONVERGENT TOOL DEFINITIONS
# ==========================================

@tool
async def navigate_to(waypoint_name: str) -> str:
    """Dispatches Argo to a specific SLAM waypoint (e.g. 'table_1', 'table_2', 'kitchen', 'docking_station', 'restrooms_waypoint')."""
    state.status = "NAVIGATING"
    state.current_waypoint = waypoint_name
    await send_to_dashboard({"command": "NAVIGATE", "target": waypoint_name})
    await send_to_dashboard({
        "event": "STATUS_UPDATE",
        "status": "NAVIGATING",
        "waypoint": waypoint_name,
    })
    return f"Navigation initiated to {waypoint_name}."


@tool
async def submit_order_to_database(items: list, table_id: str) -> str:
    """Submits the confirmed guest order list to the Kitchen Display System (KDS) database."""
    state.active_order = items
    await send_to_dashboard({
        "command": "SUBMIT_ORDER",
        "table_id": table_id,
        "items": items,
        "status": "pending",
    })
    return f"Order submitted to KDS for Table {table_id}."


@tool
async def process_refill_queue(item_name: str, table_id: str) -> str:
    """Handles water, roti, and beverage refill requests using FSM Priority Queueing."""
    if state.status in ["IDLE", "BOOT_COMPLETE"] and len(state.task_queue) == 0:
        state.status = "BUSY"
        await send_to_dashboard({"command": "NAVIGATE", "target": "kitchen"})
        return "FREE_STATUS_RESOLVED: Navigating directly to the service station to fetch the request."
    else:
        state.add_to_queue(item_name, f"table_{table_id}")
        await send_to_dashboard({
            "command": "ALERT_STAFF",
            "message": f"Table {table_id} requested {item_name}. Argo is busy.",
            "priority": "medium",
        })
        return "BUSY_STATUS_RESOLVED: Task appended to priority queue. Waitstaff alerted via KDS relay."


@tool
async def alert_floor_manager(table_id: str, reason: str) -> str:
    """Alerts the restaurant floor manager for priority human assistance."""
    await send_to_dashboard({
        "command": "ALERT_MANAGER",
        "table_id": table_id,
        "reason": reason,
        "priority": "high",
    })
    return f"Floor manager alerted for Table {table_id}: {reason}."


tools = [navigate_to, submit_order_to_database, process_refill_queue, alert_floor_manager]
tool_map = {t.name: t for t in tools}

llm = None
llm_with_tools = None


def _ensure_llm():
    global llm, llm_with_tools
    if llm is None:
        _require_api_key()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 4. DETERMINISTIC SEMANTIC PRE-ROUTER
# ==========================================

def triaged_semantic_router(user_input: str) -> str:
    clean_input = user_input.lower().strip()

    # 1. WAKE-WORD ACTIVATION (Triggers only when "Argo" is addressed explicitly)
    activation_patterns = [
        r"\b(hey argo|hello argo|hi argo|ok argo|okay argo)\b",
        r"^argo$"
    ]
    is_addressing_argo = any(re.search(pattern, clean_input) for pattern in activation_patterns)

    if not state.is_activated:
        if is_addressing_argo:
            state.is_activated = True
            state.last_speech_time = time.time()
            state.failed_stt_attempts = 0
            state.silence_prompt_count = 0
            return state.get_time_of_day_greeting()
        else:
            return None

    # 2. ACTIVE SESSION GREETINGS (Processed only if already activated)
    if state.is_activated:
        general_greetings = r"\b(hello|hi|hey|howdy|greetings|yo|sup|good morning|good afternoon|good evening)\b"
        if is_addressing_argo or re.search(general_greetings, clean_input):
            state.last_speech_time = time.time()
            return "Hello. Please let me know how I can help you."

        # 3. Creator Details FAQ
        creator_patterns = r"\b(who made you|who created you|who is your creator|who built you)\b"
        if re.search(creator_patterns, clean_input):
            state.last_speech_time = time.time()
            return "I was developed by our engineering team to help assist guests and find my way around the restaurant."

        # 4. Status FAQ
        status_patterns = r"\b(how are you|how's it going|how are you doing)\b"
        if re.search(status_patterns, clean_input):
            state.last_speech_time = time.time()
            return "I am doing well, thank you. How can I assist you at your table today?"

        # 5. Restroom wayfinding
        if re.search(r"\b(restroom|restrooms|bathroom|washroom|toilet|handwash)\b", clean_input):
            state.last_speech_time = time.time()
            return "Our restrooms are located down the main hallway, past the kitchen on your left. I hope that helps."

    return None

# ==========================================
# 5. THE AGENTIC INTERACTION LOOP
# ==========================================

system_prompt = """
ROLE: You are Argo, a poised, courteous, and highly professional restaurant service assistant.
CURRENT LOCATION: {current_waypoint}
PENDING TASKS IN QUEUE: {queue_len}

BEHAVIORAL PHILOSOPHY:
- You represent an elite hospitality standard. Speak with the grace, clarity, and attentiveness of a professional butler or dining room host.
- Your responses must be warm, reassuring, and completely free of technical, robotic, or system-level language.
- Restrict all verbal output to a maximum of two concise sentences. Do not use filler words. Be precise and elegant.
- Never refer to your system, programming, or SLAM navigation. Refer to physical movement naturally as "heading to," "visiting," or "walking over."

SITUATIONAL FLOW:

[CASE 1: GREETINGS & AMBIENT CHAT]
- Input: Guest offers a general greeting.
- Response: "Good day. Please let me know how I can help you today."

[CASE 2: DIRECT MOVEMENT REQUESTS]
- Input: Guest asks you to go somewhere specific (e.g., "Go to Table 1", "Come to Table 2", "Go to the kitchen", "Go home").
- Action: Immediately call 'navigate_to'. Map:
  * "Table X" -> 'table_X'  |  "Kitchen" -> 'kitchen'  |  "Dock"/"Home"/"Station" -> 'docking_station'
- Response: "Certainly, I am heading to Table 2 now."

[CASE 3: IMPLICIT OR UNCLEAR MOVEMENT]
- Input: Guest asks you to move without specifying a location (e.g., "come here", "please move over").
- Action: Do NOT navigate yet. Ask for clarification.
- Response: "Certainly. May I ask which table you are seated at?"

[CASE 4: ORDERING]
- When a guest places an order: "Certainly. I have recorded [Menu Items] for Table [X]. Shall I submit this order directly to the kitchen?"
- Upon confirmation: call 'submit_order_to_database', then respond: "Excellent. Your order has been submitted to the chef. I will return once it is prepared."
- If the guest updates the order mid-way: "Understood. I have updated your order to [Menu Items] for Table [X]. Shall I submit this to the kitchen now?"

[CASE 5: OUT OF STOCK]
- If an item is unavailable: "My apologies, but our [Item] is currently unavailable today. I have removed it from your order. Shall I submit the rest of your selections to the kitchen?"
- Never suggest substitutes. Simply apologize and remove the item.

[CASE 6: REFILLS — WATER / ROTI / BEVERAGES]
- When a guest asks for a refill: immediately call 'process_refill_queue'.
- If tool returns FREE_STATUS_RESOLVED: "Certainly. I will fetch a [Water / Roti] for Table [X] immediately."
- If tool returns BUSY_STATUS_RESOLVED: "Certainly. I have queued your request for Table [X] and notified our service staff who can reach you sooner."

[CASE 7: STAFF / MANAGER CALLS]
- When a guest requests a manager or staff: call 'alert_floor_manager', then respond:
  "I have sent an immediate alert to our floor manager. They will be with you at Table [X] momentarily."
"""

async def run_conversational_engine(user_input: str, chat_history: list) -> tuple:
    _ensure_llm()
    active_prompt = system_prompt.format(
        current_waypoint=state.current_waypoint,
        queue_len=len(state.task_queue),
    )

    messages = [SystemMessage(content=active_prompt)] + chat_history + [HumanMessage(content=user_input)]
    response = await llm_with_tools.ainvoke(messages)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            if tool_name in tool_map:
                print(f"[ACTION] Triggering: {tool_name} with parameters: {tool_args}")
                tool_output = await tool_map[tool_name].ainvoke(tool_args)
                # Append AIMessage (with tool_calls) and the ToolMessage result
                messages.append(response)
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))
                response = await llm_with_tools.ainvoke(messages)
                
    return response.content, chat_history + [HumanMessage(content=user_input), AIMessage(content=response.content)]

# ==========================================
# 6. SILENCE TIMEOUT MONITOR (ONE-TO-ONE ONLY)
# ==========================================

async def background_monitoring_heartbeat():
    """
    Monitors active conversation silence.
    - At 20 seconds of silence, prompts the guest ONCE.
    - At 40 seconds of silence, alerts the manager and resets to standby.
    """
    while True:
        await asyncio.sleep(1.0)
        if state.is_activated:
            elapsed = time.time() - state.last_speech_time
            
            # First Prompt at 20 seconds
            if 20.0 <= elapsed < 40.0 and state.silence_prompt_count == 0:
                print("\n[INACTIVITY] 20 seconds of silence. Prompting guest once.")
                state.silence_prompt_count = 1
                state.last_speech_time = time.time()  # Reset timer to avoid spamming
                timeout_response = "Pardon me. Please let me know how I can assist you, or which table you would like me to visit."
                
                payload = {"event": "SPEECH_RESPONSE", "text": timeout_response}
                await send_to_dashboard(payload)
                for client in list(connected_clients):
                    try:
                        await client.send(json.dumps(payload))
                    except Exception:
                        pass
            
            # Manager escalation at 40 seconds, then reset to Standby
            elif elapsed >= 40.0 and state.silence_prompt_count == 1:
                print("\n[INACTIVITY] 40 seconds of silence. Escalating to manager and resetting.")
                state.is_activated = False
                state.silence_prompt_count = 0
                state.last_speech_time = time.time()
                
                timeout_response = "As I haven't received a request, I will notify our manager to assist you at your table if needed."
                payload = {"event": "SPEECH_RESPONSE", "text": timeout_response}
                
                # Dispatch alert to custom dashboard/UI
                await send_to_dashboard({"event": "MANAGER_ALERT", "reason": "USER_SILENCE"})
                await send_to_dashboard(payload)
                
                for client in list(connected_clients):
                    try:
                        await client.send(json.dumps(payload))
                    except Exception:
                        pass

# ==========================================
# 7. WEBSOCKET SERVER INITIALIZATION
# ==========================================

async def handle_connection(websocket):
    connected_clients.add(websocket)
    print("\n[SERVER] Voice/Client connection established.")
    await send_to_dashboard({
        "event": "STATUS_UPDATE",
        "status": state.status,
        "waypoint": state.current_waypoint,
        "connected": True,
    })

    chat_history = []

    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                data = {"text": raw_message}

            control_event = data.get("event")
            if control_event == "ADMIN_START":
                state.is_activated = True
                state.last_speech_time = time.time()
                state.failed_stt_attempts = 0
                state.silence_prompt_count = 0
                print("[ADMIN] Communication session activated.")
                await send_to_dashboard({
                    "event": "STATUS_UPDATE",
                    "status": "ACTIVE",
                    "waypoint": state.current_waypoint,
                    "connected": True,
                })
                # Push wake signal to all voice clients (not back to the sender)
                for client in list(connected_clients):
                    if client != websocket:
                        try:
                            await client.send(json.dumps({"event": "ADMIN_WAKE"}))
                        except Exception:
                            pass
                continue
            if control_event == "ADMIN_STOP":
                state.is_activated = False
                state.silence_prompt_count = 0
                print("[ADMIN] Communication session deactivated.")
                await send_to_dashboard({
                    "event": "STATUS_UPDATE",
                    "status": "STANDBY",
                    "waypoint": state.current_waypoint,
                    "connected": True,
                })
                # Push sleep signal to all voice clients (not back to the sender)
                for client in list(connected_clients):
                    if client != websocket:
                        try:
                            await client.send(json.dumps({"event": "ADMIN_SLEEP"}))
                        except Exception:
                            pass
                continue

            user_input = data.get("text", "").strip()
            input_source = data.get("source", "voice")

            # -------------------------------------------------------
            # HANDLE VOICE CLIENT TIMEOUT SENTINEL
            # -------------------------------------------------------
            if user_input == "[TIMEOUT_NO_RESPONSE]":
                if state.is_activated:
                    print("[TIMEOUT] Voice client reported no speech within listen window.")
                    state.last_speech_time = time.time()
                    
                    if state.silence_prompt_count == 0:
                        state.silence_prompt_count = 1
                        timeout_text = "Pardon me, I didn't hear anything. Please let me know how I can assist you."
                        print(f"[TIMEOUT] Sending first prompt: {timeout_text}")
                        await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": timeout_text}))
                        await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": timeout_text})
                    else:
                        # Second consecutive timeout → go back to sleep
                        state.is_activated = False
                        state.silence_prompt_count = 0
                        sleep_text = "I'll step back for now. Call me again whenever you need assistance."
                        print(f"[TIMEOUT] Consecutive timeout. Deactivating. Response: {sleep_text}")
                        await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": sleep_text}))
                        await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": sleep_text})
                continue

            # If the microphone captured empty/unclear audio ("not understood")
            if not user_input:
                if state.is_activated:
                    state.failed_stt_attempts += 1
                    state.last_speech_time = time.time()
                    
                    if state.failed_stt_attempts == 1:
                        err_text = "Pardon me, I didn't quite catch that. Could you please repeat your request?"
                        print(f"[RETRY 1] STT Failed. Response: {err_text}")
                        await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": err_text}))
                        await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": err_text})
                    
                    elif state.failed_stt_attempts >= 2:
                        err_text = "I apologize for the difficulty. I will notify our manager to assist you at your table immediately."
                        print(f"[RETRY 2] STT Failed. Escalating to manager. Response: {err_text}")
                        await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": err_text}))
                        await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": err_text})
                        await send_to_dashboard({"event": "MANAGER_ALERT", "reason": "STT_FAILURE"})
                        
                        # Reset tracking states
                        state.failed_stt_attempts = 0
                        state.is_activated = False
                        state.silence_prompt_count = 0
                continue
                
            print(f"[INPUT] Received: '{user_input}' (Current Activation State: {state.is_activated})")
            await send_to_dashboard({
                "event": "USER_INPUT",
                "text": user_input,
                "source": input_source,
            })

            # Reset STT failure counter upon receiving clear input
            state.failed_stt_attempts = 0
            state.silence_prompt_count = 0

            # 1. Evaluate Greetings & Basic FAQs via Pre-Router
            direct_reply = triaged_semantic_router(user_input)
            
            if direct_reply:
                print(f"[ROUTER] Match: {direct_reply}")
                await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": direct_reply}))
                await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": direct_reply})
                continue

            # If the robot is not activated yet and wake word wasn't found in pre-router, ignore input
            if not state.is_activated:
                print("[STANDBY] Input ignored. Robot is not activated.")
                continue

            # Update activity timer for active session
            state.last_speech_time = time.time()

            # 2. Process active conversation via LLM and Tools (for navigation tasks)
            print("[AGENT] Processing active session instructions...")
            response_text, chat_history = await run_conversational_engine(user_input, chat_history)
            
            print(f"Argo response: {response_text}")
            await websocket.send(json.dumps({"event": "SPEECH_RESPONSE", "text": response_text}))
            await send_to_dashboard({"event": "SPEECH_RESPONSE", "text": response_text})
            
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

    except websockets.ConnectionClosed:
        print("[SERVER] Client disconnected.")
    finally:
        connected_clients.discard(websocket)
        if not connected_clients:
            await send_to_dashboard({
                "event": "STATUS_UPDATE",
                "status": state.status,
                "waypoint": state.current_waypoint,
                "connected": False,
            })

async def main():
    _require_api_key()
    host = os.environ.get("ARGO_AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("ARGO_AGENT_PORT", "8765"))
    print(f"[SERVER] Starting Argo NLP agent on ws://{host}:{port} ...")
    asyncio.create_task(background_monitoring_heartbeat())
    async with websockets.serve(handle_connection, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
