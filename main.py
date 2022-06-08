import cv2
import numpy as np
import math
from dataclasses import dataclass
import mediapipe as mp
import mouse
import pyautogui as pag

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

print(cap.get(cv2.CAP_PROP_FPS))

instructions = [
    "move finger to top left corner, then press 's'",
    "move finger to bottom left corner, then press 's'",
    "move finger to bottom right corner, then press 's'",
    "move finger to top right corner, then press 's'",
]


@dataclass
class Vector2:
    x: float
    y: float

    def mag(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def norm(self):
        return Vector2(self.x / self.mag(), self.y / self.mag())

    def tuple(self):
        return self.x, self.y

    @staticmethod
    def from_arr(arr):
        return Vector2(arr[0], arr[1])

    @staticmethod
    def zero():
        return Vector2(0, 0)


transform = []
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while len(transform) < 4 and cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape

    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand = hands.process(image)

    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated.flags.writeable = True

    # print(height, width)
    # print(hand.multi_hand_landmarks)
    converted = None

    if hand.multi_hand_landmarks:
        for hand_landmarks in hand.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            converted = (float(index_tip.x * width), float(index_tip.y * height))
            cv2.circle(annotated, (int(converted[0]), int(converted[1])), 20, (255, 0, 0))

    cv2.putText(annotated, instructions[len(transform)], (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))

    for point in transform:
        cv2.circle(annotated, (int(point[0]), int(point[1])), 2, (0, 255, 0))

    cv2.imshow("frame", annotated)

    wk = cv2.waitKey(1)
    if wk & 0xFF == ord('q'):
        break
    if wk & 0xFF == ord('s') and converted:
        transform.append(converted)
        print(transform)


transform = np.float32(transform)
target = np.float32([[0, 0], [0, 1],
                   [1, 1], [1, 0]])
aspect_ratio = (math.sqrt((transform[3][0] - transform[0][0]) ** 2 + (transform[3][1] - transform[0][1]) ** 2) /
                math.sqrt((transform[1][0] - transform[0][0]) ** 2 + (transform[1][1] - transform[0][1]) ** 2))
print(aspect_ratio)

matrix = cv2.getPerspectiveTransform(transform, target)

cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand = hands.process(image)

    converted = None

    if hand.multi_hand_landmarks:
        for hand_landmarks in hand.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            converted = (float(index_tip.x * width), float(index_tip.y * height))

    # warp = cv2.warpPerspective(image, matrix, (int(400 * aspect_ratio), 400))
    # warp.flags.writeable = True

    if converted:
        np_coord = np.array([[[converted[0], converted[1]]]], dtype='float32')
        new_coord = cv2.perspectiveTransform(np_coord, matrix)[0][0]
        new_coord = (max(0, min(1, new_coord[0])), max(0, min(1, new_coord[1])))
        print(new_coord)
        size = pag.size()
        mouse.move(int(new_coord[0] * size[0]), int(new_coord[1] * size[1]), True, 0)
        # cv2.circle(warp, (int(new_coord[0][0][0] * 400 * aspect_ratio), int(new_coord[0][0][1] * 400)), 20, (255,
        # 0, 0))

cap.release()
cv2.destroyAllWindows()
