import wave
import json
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us-0.15"
AUDIO_FILE = "input_mono.wav"

wf = wave.open(AUDIO_FILE, "rb")

if wf.getnchannels() != 1:
    raise ValueError("Audio must be mono")

if wf.getframerate() != 16000:
    print(f"Warning: Sample rate is {wf.getframerate()}, Vosk prefers 16000 Hz")

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break

    rec.AcceptWaveform(data)

result = json.loads(rec.FinalResult())

print("\n===== Vosk Result =====")
print(result.get("text", ""))
print("=======================")
