import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button

keyboard = KeyboardController()
mouse = MouseController()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Finger detection helpers ===
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def get_finger_states(hand_landmarks):
    return {
        'thumb':  finger_up(hand_landmarks, 4, 3),
        'index':  finger_up(hand_landmarks, 8, 6),
        'middle': finger_up(hand_landmarks, 12, 10),
        'ring':   finger_up(hand_landmarks, 16, 14),
        'pinky':  finger_up(hand_landmarks, 20, 18)
    }

# === Press / release helpers ===
def press_key(k): 
    try: keyboard.press(k)
    except: pass

def release_key(k):
    try: keyboard.release(k)
    except: pass

# === Gesture mapping ===
GESTURE_MAP = {
    "All fingers up": "Move Forward (W)",
    "Fist": "Move Backward (S)",
    "Index only": "Strafe Left (A)",
    "Index + Thumb": "Strafe Right (D)",
    "Thumb only": "Jump (SPACE)",
    "Pinky only": "Sneak (SHIFT)",
    "Index + Middle": "Attack (LMB)",
    "Pinky + Thumb": "Place/Use (RMB)",
    "Spread hand": "Open Inventory (E)",
    "Hotbar 1": "Slot 1",
    "Hotbar 2": "Slot 2",
    "Hotbar 3": "Slot 3",
    "Hotbar 4": "Slot 4",
    "Hotbar 5": "Slot 5",
    "Hotbar 6": "Slot 6",
    "Hotbar 7": "Slot 7",
    "Hotbar 8": "Slot 8",
    "Hotbar 9": "Slot 9",
    "Drop Item": "Q",
    "Sprint": "Ctrl"
}

cap = cv2.VideoCapture(0)
active_gesture = "None"

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # === Palm facing camera check ===
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            pinky_mcp = hand_landmarks.landmark[17]
            index_mcp = hand_landmarks.landmark[5]

            # Z-based check: wrist closer to camera than middle_mcp
            palm_facing = wrist.z < middle_mcp.z

            # Optional: horizontal check across palm
            palm_vector_x = pinky_mcp.x - index_mcp.x
            palm_vector_y = pinky_mcp.y - index_mcp.y
            if abs(palm_vector_x) < abs(palm_vector_y):
                palm_facing = False  # palm likely sideways

            if palm_facing:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                state = get_finger_states(hand_landmarks)
                active_gesture = "None"

                # === Movement gestures ===
                if all(state.values()):
                    press_key('w'); active_gesture = "All fingers up"
                else: release_key('w')

                if not any(state.values()):
                    press_key('s'); active_gesture = "Fist"
                else: release_key('s')

                if state['index'] and not any([state['middle'], state['ring'], state['pinky']]):
                    press_key('a'); active_gesture = "Index only"
                else: release_key('a')

                if state['index'] and state['thumb']:
                    press_key('d'); active_gesture = "Index + Thumb"
                else: release_key('d')

                if state['thumb'] and not any([state['index'], state['middle'], state['ring'], state['pinky']]):
                    press_key(Key.space); active_gesture = "Thumb only"
                else: release_key(Key.space)

                if state['pinky'] and not any([state['thumb'], state['index'], state['middle'], state['ring']]):
                    press_key(Key.shift); active_gesture = "Pinky only"
                else: release_key(Key.shift)

                # === Combat gestures ===
                if state['index'] and state['middle'] and not state['ring']:
                    mouse.press(Button.left); active_gesture = "Index + Middle"
                else: mouse.release(Button.left)

                if state['pinky'] and state['thumb']:
                    mouse.press(Button.right); active_gesture = "Pinky + Thumb"
                else: mouse.release(Button.right)

                # === Inventory ===
                if all(state.values()) and abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x) > 0.5:
                    press_key('e'); active_gesture = "Spread hand"
                else: release_key('e')

                # === Hotbar slots example (1-5) ===
                if state['pinky'] and not any([state['ring'], state['middle'], state['index']]):
                    press_key('1'); active_gesture = "Hotbar 1"
                if state['ring'] and not any([state['middle'], state['index']]):
                    press_key('2'); active_gesture = "Hotbar 2"
                if state['middle'] and not state['index']:
                    press_key('3'); active_gesture = "Hotbar 3"
                if state['index'] and not state['middle']:
                    press_key('4'); active_gesture = "Hotbar 4"
                if state['thumb'] and not any([state['index'], state['middle'], state['ring'], state['pinky']]):
                    press_key('5'); active_gesture = "Hotbar 5"

                # === Sprint example ===
                if state['index'] and state['middle'] and state['ring']:
                    press_key(Key.ctrl); active_gesture = "Sprint"
                else:
                    release_key(Key.ctrl)

            else:
                # Hand not palm-facing → ignore
                active_gesture = "None"

        # === Terminal output ===
        print(f"Gesture: {active_gesture} → {GESTURE_MAP.get(active_gesture, 'No action')}")

        # === On-screen overlay ===
        y0 = 20
        for g, action in GESTURE_MAP.items():
            color = (0, 255, 0) if g == active_gesture else (255, 255, 255)
            cv2.putText(frame, f"{g} = {action}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y0 += 20

        cv2.imshow("Minecraft Gesture Controller", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
