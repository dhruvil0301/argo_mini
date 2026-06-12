"""
Argo Voice Client — robot microphone (STT) and speaker (TTS) bridge.

Stack:
  STT  → Deepgram Nova-2 (Cloud)
  TTS  → OpenAI tts-1 (Cloud)
  I/O  → sounddevice + soundfile
  VAD  → dual-threshold RMS with fan-noise hardening + noise gate

Connects to rest_agent.py over WebSocket (port 8765).
Reports voice state to dashboard via HTTP POST /api/argo/events.
Responds to ADMIN_WAKE / ADMIN_SLEEP pushed by the agent.
"""

import asyncio
import json
import os
import re
import time
import urllib.request

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

AGENT_WS_URL  = os.environ.get("ARGO_AGENT_WS_URL", "ws://127.0.0.1:8765")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL",      "http://127.0.0.1:8080")

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

SPEECH_TIMEOUT_SLEEP  = float(os.environ.get("VOICE_WAKE_TIMEOUT",   "10.0"))
SPEECH_TIMEOUT_ACTIVE = float(os.environ.get("VOICE_LISTEN_TIMEOUT", "15.0"))

SAMPLE_RATE = 16_000
BLOCK_SIZE  = 1_024

WAKE_WORDS = ["hey argo", "hey robot", "argo"]

# Set DEBUG_VAD=1 in .env to print per-chunk RMS bars
DEBUG_VAD = os.environ.get("DEBUG_VAD", "0") == "1"

# Global speaking lock — mic callback feeds silence while Argo talks
argo_is_speaking: bool = False

# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------
def _post_dashboard(payload: dict):
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
        print(f"[Voice] Dashboard notify failed: {exc}")


def _require_api_keys():
    missing = [k for k, v in [("OPENAI_API_KEY", OPENAI_API_KEY),
                               ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY)] if not v]
    if missing:
        print(f"[Voice] ERROR: Missing API keys: {', '.join(missing)}. Add them to .env.")
        raise SystemExit(1)


def _report_state(state: str):
    print(f"[Voice] State → {state}")
    _post_dashboard({"event": "VOICE_STATE", "state": state})

# ---------------------------------------------------------------------------
# 1. ACOUSTIC UTILITIES
# ---------------------------------------------------------------------------
def calibrate_ambient_noise(duration: float = 3.0) -> tuple[float, float]:
    """Records ambient noise and returns (trigger_threshold, hold_threshold).

    Uses 90th-percentile of per-chunk RMS so noise spikes are captured,
    not just the quiet average.  Minimums raised for restaurant/fan environments.
    """
    print("\n[CALIBRATION] Measuring room noise — please remain silent for 3 seconds...")
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
    )
    sd.wait()

    # Per-chunk RMS — 90th percentile captures spike noise, not just quiet mean
    chunk_rms = [
        float(np.sqrt(np.mean(recording[i:i+BLOCK_SIZE] ** 2)))
        for i in range(0, len(recording) - BLOCK_SIZE, BLOCK_SIZE)
    ]
    noise_rms = float(np.percentile(chunk_rms, 90)) if chunk_rms else 0.008
    trigger = min(max(noise_rms * 2.8, 0.004), 0.020)
    hold = min(max(noise_rms * 2.0, 0.003), 0.010)
    # trigger = max(noise_rms * 2.8, 0.0015)   # raised: restaurant/fan ambient spikes reach 0.025–0.047
    # hold    = max(noise_rms * 2, 0.002)   # must stay above ambient so silence detection can accumulate

    print(f"[CALIBRATION] Noise 90p: {noise_rms:.4f} | Trigger: {trigger:.4f} | Hold: {hold:.4f} | RMS-only VAD")
    return trigger, hold


def play_audio(filename: str = "output.wav"):
    """Plays WAV audio with a speaking lock so the mic stays muted."""
    global argo_is_speaking
    argo_is_speaking = True
    print("🔊 [SPEAKING] Argo is speaking...")
    data, fs = sf.read(filename)
    sd.play(data, fs)
    sd.wait()   
    time.sleep(0.4)
    argo_is_speaking = False


async def record_until_silence_async(
    filename: str = "input.wav",
    trigger_threshold: float = 0.025,
    hold_threshold: float = 0.018,
    silence_limit: float = 2.0,
    speech_start_timeout: float | None = None,
) -> bool:
    """Async dual-threshold VAD recorder with fan-noise hardening.

    Returns True  → speech captured and saved to *filename* (PCM-16 WAV)
    Returns False → no speech onset, or recording was noise (discarded)
    """
    global argo_is_speaking

    audio_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _callback(indata, frames, time_arg, status):
        if argo_is_speaking:
            loop.call_soon_threadsafe(audio_queue.put_nowait, np.zeros_like(indata))
        else:
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        callback=_callback,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    )

    audio_data: list        = []
    has_spoken              = False
    silent_chunks_limit     = int((silence_limit * SAMPLE_RATE) / BLOCK_SIZE)
    silent_chunks_count     = 0
    discard_chunks_limit    = int((0.4 * SAMPLE_RATE) / BLOCK_SIZE)
    processed_chunks        = 0
    consecutive_loud        = 0
    CONFIRM_CHUNKS          = 1   # was 3 — ambient spikes rarely sustain 4 consecutive chunks
    start_time              = loop.time()
    peak_rms                = 0.0

    with stream:
        while True:
            try:
                chunk = audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                if speech_start_timeout and not has_spoken:
                    if loop.time() - start_time > speech_start_timeout:
                        if DEBUG_VAD:
                            print(f"[VAD] Timeout after {loop.time()-start_time:.1f}s "
                                  f"— no voice onset (peak={peak_rms:.4f}, trigger={trigger_threshold:.4f})")
                        return False
                continue

            processed_chunks += 1
            if processed_chunks <= discard_chunks_limit:
                continue

            audio_data.append(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > peak_rms:
                peak_rms = rms

            if DEBUG_VAD and len(audio_data) % 10 == 0:
                bar_len = int(rms / trigger_threshold * 20)
                bar     = "█" * min(bar_len, 40)
                marker  = "▶" if has_spoken else ("!" if consecutive_loud > 0 else " ")
                print(f"[VAD] {marker} rms={rms:.4f} | trig={trigger_threshold:.4f} | "
                      f"hold={hold_threshold:.4f} | consec={consecutive_loud}/{CONFIRM_CHUNKS} | "
                      f"silence={silent_chunks_count}/{silent_chunks_limit} |{bar:<20}|")

            if not has_spoken:
                if rms > trigger_threshold:
                    consecutive_loud += 1
                    if consecutive_loud == 1:
                        print(f"[VAD] Possible voice onset — rms={rms:.4f} "
                              f"(need {CONFIRM_CHUNKS} consecutive loud chunks)")
                    elif consecutive_loud > 1:
                        print(f"[VAD] Confirmation progress: {consecutive_loud}/{CONFIRM_CHUNKS} "
                              f"(rms={rms:.4f})")
                    if consecutive_loud >= CONFIRM_CHUNKS:
                        print(f"[VOICE] Voice confirmed. Recording... (peak rms so far: {peak_rms:.4f})")
                        has_spoken = True
                        silent_chunks_count = 0
                else:
                    if consecutive_loud > 0:
                        print(f"[VAD] Reset — rms dropped to {rms:.4f} "
                              f"(was {consecutive_loud} consecutive loud chunks)")
                    consecutive_loud = 0
            else:
                if rms > hold_threshold:
                    silent_chunks_count = 0
                else:
                    silent_chunks_count += 1

            if has_spoken and silent_chunks_count >= silent_chunks_limit:
                print("[SILENCE] End of speech. Stopping recording.")
                break

            if len(audio_data) * BLOCK_SIZE > SAMPLE_RATE * 15:
                print("⚠️  [SYSTEM] Max recording length (15 s). Stopping.")
                break

    full_audio = np.concatenate(audio_data, axis=0)
    duration_s = len(full_audio) / SAMPLE_RATE
    mean_rms   = float(np.sqrt(np.mean(full_audio ** 2)))

    # Noise gate: discard only when BOTH mean is very low AND peak never reached
    # speech level.  A short phrase like "Hey Argo" has low mean but clear peak.
    if mean_rms < hold_threshold * 0.8 and peak_rms < trigger_threshold * 3.5:
        print(f"[VAD] ⚠ Noise gate — mean={mean_rms:.4f} < hold*0.8={hold_threshold*0.8:.4f} "
              f"AND peak={peak_rms:.4f} < trig*3.5={trigger_threshold*3.5:.4f}. Discarding.")
        return 

    print(f"[VAD] Recording saved → '{filename}' | "
          f"duration={duration_s:.2f}s | mean_rms={mean_rms:.4f} | peak_rms={peak_rms:.4f}")
    # PCM-16 — most reliable format for cloud STT APIs
    audio_int16 = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(filename, audio_int16, SAMPLE_RATE, subtype="PCM_16")
    print(f"[MIC] duration={duration_s:.2f}s "f"mean_rms={mean_rms:.4f} "f"peak_rms={peak_rms:.4f}")
    return True

# ---------------------------------------------------------------------------
# 2. CLOUD STT — Deepgram Nova-2
# ---------------------------------------------------------------------------
def transcribe_audio(filename: str = "input.wav") -> str:
    """Sends recorded WAV to Deepgram Nova-2. Returns transcript or ''."""
    try:
        info = sf.info(filename)
        if info.duration < 0.5:
            print(f"[STT] Skipping — recording too short ({info.duration:.2f}s)")
            return ""
    except Exception:
        pass

    print(f"[STT] Transcribing '{filename}' via Deepgram...")
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "audio/wav",
    }
    try:
        with open(filename, "rb") as f:
            audio_data = f.read()

        print("=" * 60)
        print(f"[STT] Sending audio to Deepgram")
        print(f"[STT] File: {filename}")
        print(f"[STT] Size: {len(audio_data):,} bytes")

        url = (
            "https://api.deepgram.com/v1/listen"
            "?model=nova-2"
            "&language=en-US"
            "&punctuate=true"
        )

        start_time = time.time()
        response = requests.post(url, headers=headers, data=audio_data, timeout=20)
        elapsed = time.time() - start_time

        print(f"[STT] HTTP Status: {response.status_code}")
        print(f"[STT] Response Time: {elapsed:.2f}s")

        if response.status_code == 200:
            result = response.json()
            try:
                alt = result["results"]["channels"][0]["alternatives"][0]
                transcript = alt.get("transcript", "").strip()
                confidence = alt.get("confidence", 0)
                print(f"[STT] Confidence: {confidence:.3f}")
                print(f"[STT] Transcript: '{transcript}'")
                if not transcript:
                    print("[STT] ⚠️ Empty transcript returned by Deepgram")
                    print("[STT] Raw Response:")
                    print(result)
                print("=" * 60)
                return transcript
            except Exception as parse_err:
                print(f"[STT] ❌ Failed to parse Deepgram response: {parse_err}")
                print("[STT] Raw Response:")
                print(result)
                print("=" * 60)
                return ""
        else:
            print(f"[STT] ❌ Deepgram Error {response.status_code}")
            print(response.text)
            print("=" * 60)
            return ""

    except requests.exceptions.Timeout:
        print("[STT] ❌ Deepgram timeout after 20 seconds")
        return ""
    except Exception as exc:
        print(f"[STT] ❌ Deepgram API error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# 3. CLOUD TTS — OpenAI tts-1
# ---------------------------------------------------------------------------
def text_to_speech(text: str, filename: str = "output.wav"):
    """Converts text to speech using OpenAI TTS and saves as WAV."""
    clean = (text or "").strip()
    if not clean:
        return
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":           "tts-1",
        "input":           clean,
        "voice":           "alloy",
        "response_format": "wav",
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers, json=payload, timeout=20
        )
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            print(f"[TTS] ❌ OpenAI error: {response.status_code} — {response.text}")
    except Exception as exc:
        print(f"[TTS] ❌ OpenAI API error: {exc}")

# ---------------------------------------------------------------------------
# 4. WEBSOCKET HELPERS
# ---------------------------------------------------------------------------
async def _ws_recv_loop(ws, queue: asyncio.Queue):
    """Background task — forwards every server-push message into queue."""
    try:
        async for raw in ws:
            await queue.put(raw)
    except Exception:
        pass


def _pop_control_event(queue: asyncio.Queue) -> str | None:
    """Non-blocking drain: returns first ADMIN_WAKE/ADMIN_SLEEP found, puts others back."""
    stash, found = [], None
    try:
        while True:
            raw = queue.get_nowait()
            try:
                evt = json.loads(raw).get("event", "")
                if evt in ("ADMIN_WAKE", "ADMIN_SLEEP") and found is None:
                    found = evt
                else:
                    stash.append(raw)
            except Exception:
                stash.append(raw)
    except asyncio.QueueEmpty:
        pass
    for item in stash:
        queue.put_nowait(item)
    return found

# ---------------------------------------------------------------------------
# 5. MAIN VOICE CLIENT LOOP
# ---------------------------------------------------------------------------
async def run_voice_client():
    print(f"[*] Connecting to Argo agent at {AGENT_WS_URL}...")
    _report_state("connecting")

    async with websockets.connect(AGENT_WS_URL, open_timeout=10) as ws:
        print("[*] Voice client connected. Ready!")
        _report_state("idle")

        loop = asyncio.get_running_loop()

        # Dynamic room-noise calibration (blocking — run in thread)
        trigger_threshold, hold_threshold = await loop.run_in_executor(
            None, calibrate_ambient_noise, 3.0
        )

        # Background recv task queues all server-push messages
        server_queue: asyncio.Queue = asyncio.Queue()
        recv_task = asyncio.create_task(_ws_recv_loop(ws, server_queue))

        session_active = False
        timeout_count  = 0
        print("\n💤 [SLEEPING] Say 'Hey Argo', 'Hey Robot', or 'Argo' to wake me up.")

        try:
            while True:

                if not session_active:
                    # ── SLEEP MODE ───────────────────────────────────────
                    _report_state("listening_wake")
                    recorded = await record_until_silence_async(
                        filename="input.wav",
                        trigger_threshold=trigger_threshold,
                        hold_threshold=hold_threshold,
                        speech_start_timeout=SPEECH_TIMEOUT_SLEEP,
                    )
                    if not recorded:
                        # VAD timeout — check for admin wake signal
                        ctrl = _pop_control_event(server_queue)
                        if ctrl == "ADMIN_WAKE":
                            session_active = True
                            timeout_count  = 0
                            print("⚡ [ADMIN WAKE] Activated by admin panel.")
                            _report_state("active")
                        continue

                    user_text = await loop.run_in_executor(None, transcribe_audio, "input.wav")
                    if not user_text.strip():
                        continue

                    cleaned      = user_text.lower().strip()
                    wake_detected = any(
                        re.search(r"\b" + re.escape(w) + r"\b", cleaned)
                        for w in WAKE_WORDS
                    )

                    if wake_detected:
                        print(f"⚡ [WAKE WORD] Detected in: '{user_text}'. Activating session.")
                        session_active = True
                        timeout_count  = 0
                        _report_state("active")

                        await ws.send(json.dumps({"text": user_text, "source": "voice"}))
                        raw  = await server_queue.get()
                        data = json.loads(raw)
                        if data.get("event") == "SPEECH_RESPONSE":
                            reply = data.get("text", "")
                            print(f"\n🤖 Argo: '{reply}'")
                            _report_state("speaking")
                            await loop.run_in_executor(None, text_to_speech, reply, "output.wav")
                            await loop.run_in_executor(None, play_audio, "output.wav")
                            _report_state("listening")
                    else:
                        print(f"💤 [SLEEPING] Heard: '{user_text}' — no wake word.")

                else:
                    # ── ACTIVE SESSION ────────────────────────────────────
                    # Check for admin sleep command before each listen cycle
                    ctrl = _pop_control_event(server_queue)
                    if ctrl == "ADMIN_SLEEP":
                        session_active = False
                        timeout_count  = 0
                        print("😴 [ADMIN SLEEP] Admin panel put Argo to sleep.")
                        _report_state("idle")
                        print("\n💤 [SLEEPING] Say 'Hey Argo' to wake me up again.")
                        continue

                    print("\n[LISTENING] Argo is listening... Speak now!")
                    _report_state("listening")

                    recorded = await record_until_silence_async(
                        filename="input.wav",
                        trigger_threshold=trigger_threshold,
                        hold_threshold=hold_threshold,
                        silence_limit=2.0,
                        speech_start_timeout=SPEECH_TIMEOUT_ACTIVE,
                    )

                    user_text = ""
                    if recorded:
                        _report_state("processing")
                        user_text = await loop.run_in_executor(None, transcribe_audio, "input.wav")
                        user_text = user_text.strip()

                    # ── Silence / STT failure ─────────────────────────────
                    if not recorded or not user_text:
                        if timeout_count == 0:
                            timeout_count += 1
                            print("\n⏳ [TIMEOUT] No speech detected. Notifying agent...")
                            await ws.send(json.dumps({"text": "[TIMEOUT_NO_RESPONSE]", "source": "voice"}))
                            raw  = await server_queue.get()
                            data = json.loads(raw)
                            if data.get("event") == "SPEECH_RESPONSE":
                                reply = data.get("text", "")
                                print(f"\n🤖 Argo: '{reply}'")
                                _report_state("speaking")
                                await loop.run_in_executor(None, text_to_speech, reply, "output.wav")
                                await loop.run_in_executor(None, play_audio, "output.wav")
                            continue
                        else:
                            print("\n⏳ [TIMEOUT] Consecutive timeout. Returning to sleep.")
                            session_active = False
                            timeout_count  = 0
                            _report_state("idle")
                            print("\n💤 [SLEEPING] Say 'Hey Argo' to wake me up again.")
                            continue

                    # ── Good transcription ────────────────────────────────
                    timeout_count = 0
                    print(f"👉 You said: '{user_text}'")

                    cleaned = user_text.lower().strip()
                    if any(cmd in cleaned for cmd in
                           ["goodbye", "exit", "quit", "thank you", "bye", "go to sleep"]):
                        print("👋 [EXIT] Exit command. Going to sleep after response.")
                        session_active = False

                    await ws.send(json.dumps({"text": user_text, "source": "voice"}))
                    _report_state("waiting_reply")

                    raw  = await server_queue.get()
                    data = json.loads(raw)
                    if data.get("event") == "SPEECH_RESPONSE":
                        reply = data.get("text", "")
                        print(f"\n🤖 Argo: '{reply}'")
                        _report_state("speaking")
                        await loop.run_in_executor(None, text_to_speech, reply, "output.wav")
                        await loop.run_in_executor(None, play_audio, "output.wav")
                        _report_state("listening" if session_active else "idle")
                        if not session_active:
                            await ws.send(json.dumps({"event": "ADMIN_STOP"}))
                            print("[EXIT] ADMIN_STOP sent — agent session deactivated.")

        finally:
            recv_task.cancel()
            try:
                await asyncio.wait_for(recv_task, timeout=1.0)
            except Exception:
                pass


async def main():
    print("[Voice] Starting Argo voice client (Cloud STT + TTS)...")
    _require_api_keys()
    while True:
        try:
            await run_voice_client()
        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as exc:
            print(f"[Voice] Connection lost: {exc}. Retrying in 3 s...")
            _report_state("offline")
            await asyncio.sleep(3.0)
        except KeyboardInterrupt:
            print("\n[*] Exiting voice client.")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Exiting Voice Client.")
