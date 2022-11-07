# import os
# import psutil
# import time
#
# import requests
#
# p1, p5, p15 = psutil.getloadavg()
# # .Process(os.getpid())
# # print(psutil.cpu_percent(1))
# cpu_usage = (p15 / os.cpu_count()) * 100
# # time.sleep(3)
# print("The CPU usage is : ", cpu_usage)
#
# # import cv2
# # import numpy as np
# # url = "http://192.168.18.150:8080/video"
# # cap = cv2.VideoCapture(url)
# # cap.set(3, 640)
# # cap.set(4, 480)
# # while True:
# #     # img_resp = requests.get(url)
# #     # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# #     # imgOriginalScene = cv2.imdecode(img_arr, -1)
# #     camera, frame = cap.read()
# #     print(camera)
# #     if frame is not None:
# #         cv2.imshow("Frame", frame)
# #     q = cv2.waitKey(1)
# #     if q == ord("q"):
# #         break
# # cv2.destroyAllWindows()
# #     print(camera)##
# #     if frame is not None:
# #         cv2.imshow("Frame", frame)
# #     q = cv2.waitKey(1)##
# #     if q == ord("q"):
# #         break ###
# # cv2.destroyAllWindows()##
#
#
# import tkinter as tk
#
# root = tk.Tk()
#
# canvas1 = tk.Canvas(root, width=300, height=300)
# canvas1.pack()
#
#
# def hello():
#     label1 = tk.Label(root, text='Hello World!', fg='blue', font=('helvetica', 12, 'bold'))
#     canvas1.create_window(150, 200, window=label1)
#
#
# button1 = tk.Button(text='Click Me', command=hello, bg='brown', fg='white')
# canvas1.create_window(150, 150, window=button1)
#
# root.mainloop()
import time

import pyautogui
# import cv2
# import mediapipe as mp
# import time
#
# cap = cv2.VideoCapture(0)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# cTime = 0
# pTime = 0
# # Function Start
# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     # print(results.multi_hand_landmarks)
#     if results.multi_hand_landmarks:
#         for handlms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handlms.landmark):
#                 # print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 print(id, cx, cy)
#                 # if id == 5:
#                 # cv2.circle(img, (cx, cy), 15, (139, 0, 0), cv2.FILLED)
#
#             mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
#     # Time and FPS Calculation
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139, 0, 0), 3)
#
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# import easyocr
#
# reader = easyocr.Reader(['en'])
# result = reader.readtext('1.png')
# mytxt = " ".join(([r[1] for r in result]))
# open("converted.txt", "w").write(mytxt)


# import pyautogui
# import time
# time.sleep(5)
# c = 1
# while c <= 10:
#     pyautogui.typewrite("It's me A")
#     pyautogui.press("enter")
#     c+=1


# from io import BytesIO
# import ffmpeg
# # data = ""
# with open('demovid.txt', 'rb') as file:
#     f = BytesIO(file.read())
#     process = (
#         ffmpeg.input('pipe:').output('test.mp4').overwrite_output().run_async(pipe_stdin=True)
#     )
#     process.communicate(input=f.getbuffer())

# print(open("test.mp4", "rb").read() == open("demovid.txt", "rb").read())


name = "Hassaan Ahmad"
name = list(name)
ix = name.index(" ")
for i in range(ix//2):
    name[i], name[ix-1-i] = name[ix-1-i], name[i]
j = -1
for i in range(ix+1, (len(name)+ix+1)//2):
    name[i], name[j] = name[j], name[i]
    j-=1
print("".join(name))

from hand_detection import hand_detect
import cv2
t = time.time()
img = cv2.imread("face_recognition/k.jpg")
print(hand_detect(img))

print(time.time()-t)
