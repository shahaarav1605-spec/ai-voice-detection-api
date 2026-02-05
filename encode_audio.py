import base64

audio_path = r"D:\project1\data\human\tamil\sample.wav"

with open(audio_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print(encoded)
