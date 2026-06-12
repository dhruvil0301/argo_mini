import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    print(f"Level: {np.abs(indata).mean():.4f}")

with sd.InputStream(
    device=37,
    channels=2,
    samplerate=16000,
    callback=callback
):
    input("Speak and press Enter to stop\n")
