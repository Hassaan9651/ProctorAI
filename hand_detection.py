# Importing Libraries
import cv2
import mediapipe as mp

# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    # model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)


def hand_detect(img):
    # Initializing the Model

    # Start capturing video from webcam
    # cap = cv2.VideoCapture(0)

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 0:
            # Display 'Both Hands' on the image
            return "No Hands Detected"

        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            return "Both Hands Detected"
        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    # Display 'Left Hand' on
                    # left side of window
                    return "Left Hand Detected"
                if label == 'Right':
                    return "Right Hand Detected"

                # Display 'Left Hand'
                # on left side of window
