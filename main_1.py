import cv2
import numpy as np
import time
from tensorflow import keras

m = keras.models.load_model("handmodel_fingers_model1.h5")

u = False      #hand_raised or not
v = 0       #track of the number of times the finger is raised
w = False   # whether the finger is currently raised or not
x = None  #start time when the finger is raised

def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0  
    return frame

c = cv2.VideoCapture(0)
d = 10

while True:
    r, frame = c.read()

    if not r:
        break

    p = preprocess_frame(frame)
    i = np.expand_dims(p, axis=0)
    i = i.reshape((1, 128, 128, 1))
    y = m.predict(i)
    z = np.argmax(y)  #finger_number

    if z > 2:
        u = True
    else:
        u = False

    if z > 0 and not w:
        w = True
        x = time.time()
    elif z == 0:
        w = False
        x = None
    elif x is not None and time.time() - x >= 3:
        v += 1
        x = None

    tc = (0, 255, 0) if u else (0, 0, 255)

    cv2.putText(frame, "Fingers: " + str(z), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Raise Count: " + str(v), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Hand Raised: ", (frame.shape[1] - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, tc, 2)

    cv2.imshow("Finger Detection", frame)

    dt = 1 / d
    time.sleep(dt)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

c.release()
cv2.destroyAllWindows()
