import os
import psutil
import time

import requests

p1, p5, p15 = psutil.getloadavg()
        # .Process(os.getpid())
# print(psutil.cpu_percent(1))
cpu_usage = (p15 / os.cpu_count()) * 100
# time.sleep(3)
print("The CPU usage is : ", cpu_usage)

# import cv2
# import numpy as np
# url = "http://192.168.18.150:8080/video"
# cap = cv2.VideoCapture(url)
# cap.set(3, 640)
# cap.set(4, 480)
# while True:
#     # img_resp = requests.get(url)
#     # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     # imgOriginalScene = cv2.imdecode(img_arr, -1)
#     camera, frame = cap.read()
#     print(camera)##
#     if frame is not None:
#         cv2.imshow("Frame", frame)
#     q = cv2.waitKey(1)##
#     if q == ord("q"):
#         break ###
# cv2.destroyAllWindows()##
