import requests
import time

data = {
    "name": "Hussey",
    "video_chunk": open("video.txt", 'r').read(),
    "ai_logs": "None"
}
t = time.time()
# response = requests.get("http://127.0.0.1:8000/")

response = requests.post("http://127.0.0.1:8000/vision/", json=data)
print("Response Time: ", time.time() - t)
print(response.json())
