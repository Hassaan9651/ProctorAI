import math
import os
import time

import psutil

import face_landmarks as fl
import face_detector as fd
import tensorflow as tf
import cv2
import numpy as np
from head_pose_estimation import get_2d_points, draw_annotation_box, head_pose_points
from eye_tracker import eye_on_mask, find_eyeball_position, contouring, process_thresh, print_eye_pos
import person_and_phone as pnp
from datetime import datetime

import uuid

# import tkinter as tk

# root = tk.Tk()
#
# canvas1 = tk.Canvas(root, width=300, height=300)
# canvas1.pack()
face_model = fd.get_face_detector()
landmark_model = fl.get_landmark_model()
yolo = pnp.YoloV3()

pnp.load_darknet_weights(yolo, 'models/yolov3.weights')
class_names = [c.strip() for c in open("models/classes.TXT").readlines()]

def proctor(video_path=0):
    # label1 = tk.Label(root, text='Hello World!', fg='blue', font=('helvetica', 12, 'bold'))
    # canvas1.create_window(150, 200, window=label1)

    # print(tf.config.list_physical_devices('GPU'))

    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0] * 5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0] * 3

    macid = uuid.getnode()
    cap = cv2.VideoCapture(video_path)

    # url = "http://192.168.18.150:8080/video"
    # cap = cv2.VideoCapture(url)
    # cap.set(3, 640)
    # cap.set(4, 480)

    valid, img = cap.read()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    thresh = img.copy()

    while True:
        ret, img = cap.read()
        rects = fd.find_faces(img, face_model)
        for rect in rects:
            shape = fl.detect_marks(img, landmark_model, rect)
            # fl.draw_marks(img, shape)
            cv2.putText(img, "Press 's' to start AI-Proctor", (30, 30), font,
                        1, (0, 255, 255), 2)
            cv2.imshow("CAM", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("shit")
            exit(0)
        elif True:
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break

    cv2.destroyAllWindows()
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]

    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    cv2.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    def nothing(x):
        pass

    cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

    file = open("Monitoring_report.csv", "w")
    file.write("Mac Id, TimeStamp, Model Name, Event Details, CPU Usage(%), RAM(%)\n")

    i = 0
    while True:
        valid, img = cap.read()
        p1, p5, p15 = psutil.getloadavg()
        cpu_usage = (p1 / os.cpu_count()) * 100
        vram = psutil.virtual_memory()[2]
        if valid:
            if i % 10 == 0:
                # continue
                # --------------------YOLO Section-------------------
                # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img1 = cv2.resize(img, (320, 320))
                # img1 = img1.astype(np.float32)
                # img1 = np.expand_dims(img1, 0)
                # img1 = img1 / 255
                # # class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
                # t = time.time()
                # boxes, scores, classes, nums = yolo(img1)
                # print(time.time() - t)
                # count = 0
                # for i in range(nums[0]):
                #     if int(classes[0][i] == 0):
                #         count += 1
                #     if int(classes[0][i] == 67):
                #         msg = str(macid) + ', ' + str(datetime.now())[
                #                                   :-7] + ', Object Detection, Mobile Phone detected, ' + str(
                #             cpu_usage) + ', ' + str(vram) + '\n'
                #         print(msg)
                #         file.write(msg)
                #         # print('Mobile Phone detected')
                # if count == 0:
                #     print('No person detected')
                #     msg = str(macid) + ', ' + str(datetime.now())[
                #                               :-7] + ', Object Detection, No person detected, ' + str(
                #         cpu_usage) + ', ' + str(vram) + '\n'
                #     print(msg)
                #     file.write(msg)
                # elif count > 1:
                #     msg = str(macid) + ', ' + str(datetime.now())[
                #                               :-7] + ', Object Detection, More than one person detected, ' + str(
                #         cpu_usage) + ', ' + str(vram) + '\n'
                #     print(msg)
                #     file.write(msg)
                #     # print('More than one person detected')
                #
                # img = pnp.draw_outputs(img, (boxes, scores, classes, nums), class_names)

                # -----------------NON YOLO Section-----------------
                faces = fd.find_faces(img, face_model)
                if len(faces) > 1:
                    msg = str(macid) + ', ' + str(datetime.now())[:-7] + ', Face Detection, ' + str(
                        len(faces)) + ' faces detected, ' + str(
                        cpu_usage) + ', ' + str(vram) + '\n'
                    print(msg)
                    file.write(msg)
                    # print(len(faces), "faces detected!")

                fd.draw_faces(img, faces)

                for face in faces:
                    marks = fl.detect_marks(img, landmark_model, face)

                    # Mouth Opening Detection -------------start-----------------
                    cnt_outer = 0
                    cnt_inner = 0
                    fl.draw_marks(img, marks[48:])
                    for i, (p1, p2) in enumerate(outer_points):
                        if d_outer[i] + 3 < marks[p2][1] - marks[p1][1]:
                            cnt_outer += 1
                    for i, (p1, p2) in enumerate(inner_points):
                        if d_inner[i] + 2 < marks[p2][1] - marks[p1][1]:
                            cnt_inner += 1
                    if cnt_outer > 3 and cnt_inner > 2:
                        msg = str(macid) + ', ' + str(datetime.now())[
                                                  :-7] + ', Mouth Opening Detection, Mouth Opened, ' + str(
                            cpu_usage) + ', ' + str(vram) + '\n'
                        print(msg)
                        file.write(msg)
                        cv2.putText(img, 'Mouth open', (30, 30), font,
                                    1, (0, 255, 255), 2)
                    # Mouth Opening Detection -------------End-----------------

                    # Eye Tracking -------------start-----------------

                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    mask, end_points_left = eye_on_mask(mask, left, marks)
                    mask, end_points_right = eye_on_mask(mask, right, marks)
                    mask = cv2.dilate(mask, kernel, 5)

                    eyes = cv2.bitwise_and(img, img, mask=mask)
                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]
                    mid = int((marks[42][0] + marks[39][0]) // 2)
                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                    thresh = process_thresh(thresh)

                    eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                    eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                    eye_activity = print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
                    if eye_activity:
                        msg = str(macid) + ', ' + str(datetime.now())[
                                                  :-7] + ', Eye Tracking, ' + eye_activity + ", " + str(
                            cpu_usage) + ', ' + str(vram) + '\n'
                        print(msg)
                        file.write(msg)
                    # Eye Tracking -------------End-----------------

                    # head pose estimation -------------start-----------------
                    # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                    image_points = np.array([
                        marks[30],  # Nose tip
                        marks[8],  # Chin
                        marks[36],  # Left eye left corner
                        marks[45],  # Right eye right corne
                        marks[48],  # Left Mouth corner
                        marks[54]  # Right mouth corner
                    ], dtype="double")
                    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                                  camera_matrix,
                                                                                  dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose

                    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                     translation_vector, camera_matrix, dist_coeffs)

                    for p in image_points:
                        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                    # cv2.line(img, p1, p2, (0, 255, 255), 2)
                    # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

                    # for (x, y) in marks:
                    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                    try:
                        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                        ang1 = int(math.degrees(math.atan(m)))
                    except:
                        ang1 = 90

                    try:
                        m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                        ang2 = int(math.degrees(math.atan(-1 / m)))
                    except:
                        ang2 = 90

                        # print('div by zero error')
                    if ang1 >= 48:
                        msg = str(macid) + ', ' + str(datetime.now())[:-7] + ', Head Pose, Head Turned Down, ' + str(
                            cpu_usage) + ', ' + str(
                            vram) + '\n'
                        print(msg)
                        file.write(msg)
                        cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                    elif ang1 <= -48:
                        msg = str(macid) + ', ' + str(datetime.now())[:-7] + ', Head Pose, Head Turned up, ' + str(
                            cpu_usage) + ', ' + str(
                            vram) + '\n'
                        print(msg)
                        file.write(msg)
                        cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

                    if ang2 >= 48:
                        msg = str(macid) + ', ' + str(datetime.now())[:-7] + ', Head Pose, Head Turned Right, ' + str(
                            cpu_usage) + ', ' + str(
                            vram) + '\n'
                        print(msg)
                        file.write(msg)
                        cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                    elif ang2 <= -48:
                        msg = str(macid) + ', ' + str(datetime.now())[:-7] + ', Head Pose, Head Turned left, ' + str(
                            cpu_usage) + ', ' + str(
                            vram) + '\n'
                        print(msg)
                        file.write(msg)
                        cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                    # To show angles on feed uncomment
                    # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                    # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
                    # head pose estimation -------------End-----------------
                cv2.imshow('AI Proctor', img)
                cv2.imshow("Eye Map", thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        i += 1

    file.close()
    cv2.destroyAllWindows()
    cap.release()

# button1 = tk.Button(text='Click Me', command=hello, bg='brown', fg='white')
# canvas1.create_window(150, 150, window=button1)
#
# root.mainloop()

# proctor(video_path="test.mp4")
