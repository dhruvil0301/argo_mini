"""
Microphone test — records 5 seconds, shows live RMS, then transcribes via Deepgram.
Run: python3 mic_test.py
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
SAMPLE_RATE      = 16_000
DURATION         = 5       # seconds to record
OUTPUT_FILE      = "mic_test.wav"

# ── 1. List audio devices ────────────────────────────────────────────────────
print("=" * 60)
print("AVAILABLE AUDIO INPUT DEVICES")
print("=" * 60)
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d["max_input_channels"] > 0:
        marker = " ← default" if i == sd.default.device[0] else ""
        print(f"  [{i:2d}] {d['name']}  (channels: {d['max_input_channels']}){marker}")

default_in = sd.default.device[0]
print(f"\nUsing device [{default_in}]: {sd.query_devices(default_in)['name']}")
print("=" * 60)

# ── 2. Record with live RMS bar ───────────────────────────────────────────────
print(f"\nRecording {DURATION}s — speak into the microphone now...\n")

frames = []

def callback(indata, frame_count, time_info, status):
    frames.append(indata.copy())
    rms = float(np.sqrt(np.mean(indata ** 2)))
    bar_len = min(int(rms * 400), 50)
    bar = "█" * bar_len
    level = "LOUD" if rms > 0.05 else ("OK" if rms > 0.015 else "quiet")
    print(f"\r  RMS: {rms:.4f}  |{bar:<50}|  {level}   ", end="", flush=True)

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
    sd.sleep(DURATION * 1000)

print("\n")

if not frames:
    print("❌ No audio captured. Check microphone connection.")
    sys.exit(1)

audio = np.concatenate(frames, axis=0)
mean_rms = float(np.sqrt(np.mean(audio ** 2)))
peak_rms = float(np.max(np.abs(audio)))

print(f"  Duration : {len(audio)/SAMPLE_RATE:.2f}s")
print(f"  Mean RMS : {mean_rms:.4f}")
print(f"  Peak     : {peak_rms:.4f}")

if mean_rms < 0.001:
    print("\n⚠️  Mean RMS is near zero — microphone may be muted or disconnected.")
elif mean_rms < 0.005:
    print("\n⚠️  Signal is very low — try increasing mic gain with: alsamixer")
else:
    print("\n✅ Audio captured successfully.")

# Save PCM-16 WAV (best for STT)
audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("int16")
sf.write(OUTPUT_FILE, audio_int16, SAMPLE_RATE, subtype="PCM_16")
print(f"  Saved    : {OUTPUT_FILE}")

# ── 3. Transcribe via Deepgram ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEEPGRAM TRANSCRIPTION")
print("=" * 60)

if not DEEPGRAM_API_KEY:
    print("❌ DEEPGRAM_API_KEY not set in .env — skipping STT test.")
    sys.exit(0)

print("Sending to Deepgram Nova-2...")
try:
    with open(OUTPUT_FILE, "rb") as f:
        audio_bytes = f.read()

    url = "https://api.deepgram.com/v1/listen?model=nova-2&language=en-US&punctuate=true"
    resp = requests.post(
        url,
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
        data=audio_bytes,
        timeout=15,
    )

    if resp.status_code == 200:
        transcript = resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        confidence = resp.json()["results"]["channels"][0]["alternatives"][0].get("confidence", 0)
        if transcript.strip():
            print(f"\n✅ Transcript  : \"{transcript}\"")
            print(f"   Confidence  : {confidence:.2f}")
        else:
            print("\n⚠️  Deepgram returned empty transcript.")
            print("   Possible causes:")
            print("   - Recording was silence / background noise only")
            print("   - Mic gain too low (try: alsamixer)")
            print("   - Wrong microphone device selected")
    else:
        print(f"\n❌ Deepgram error {resp.status_code}: {resp.text}")

except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
print("Test complete. Check mic_test.wav to listen back.")
print("=" * 60)
