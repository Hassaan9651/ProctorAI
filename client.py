import base64

import requests
import time
import numpy as np

data = {
    "name": "Hussey",
    "video_chunk": open("ronaldo.txt", 'r').read(),
    # "video_chunk": base64.b64encode(open("test.mp4", 'rb').read()),
    "ai_logs": "None"
}

times = []
tests = 1
for i in range(tests):
    t = time.time()
    # response = requests.get("http://127.0.0.1:8000/")

    response = requests.post("http://127.0.0.1:8000/vision/", json=data)
    responseTime = time.time() - t
    times.append(responseTime)
    print(i+1)
print("Response Time Avg(", tests, "): ", np.array(times).mean())
print(response.json())
