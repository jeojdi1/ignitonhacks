import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
from collections import deque

keyboard = KeyboardController()
mouse = MouseController()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Finger detection helpers ===
def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def thumb_up(hand_landmarks, hand_label="Right"):
    tip = hand_landmarks.landmark[4]
    mcp = hand_landmarks.landmark[2]
    if hand_label == "Right":
        return tip.x < mcp.x  # Right thumb sticks left
    else:
        return tip.x > mcp.x  # Left thumb sticks right

def get_finger_states(hand_landmarks, hand_label="Right"):
    return {
        'thumb':  thumb_up(hand_landmarks, hand_label),
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

# === Gesture buffer for stabilization ===
buffer_length = 3
gesture_buffers = {'Left': deque(maxlen=buffer_length), 'Right': deque(maxlen=buffer_length)}

cap = cv2.VideoCapture(0)
window_name = "Minecraft Gesture Controller"

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gestures_detected = {'Left': "None", 'Right': "None"}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_handedness.classification[0].label  # "Left" or "Right"
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                pinky_mcp = hand_landmarks.landmark[17]
                index_mcp = hand_landmarks.landmark[5]

                # Palm facing detection
                z_threshold = 0.05
                palm_facing = (middle_mcp.z - wrist.z) > -z_threshold
                palm_vector_x = pinky_mcp.x - index_mcp.x
                palm_vector_y = pinky_mcp.y - index_mcp.y
                if abs(palm_vector_x) < abs(palm_vector_y):
                    palm_facing = False

                if not palm_facing:
                    continue

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                state = get_finger_states(hand_landmarks, hand_label)
                gesture_name = "None"

                if hand_label == "Right":
                    # Right hand → movement, jump, sneak, inventory, sprint, placement
                    if all(state.values()):
                        gesture_name = "Move Forward"; press_key('w')
                    else: release_key('w')

                    if not any(state.values()):
                        gesture_name = "Move Backward"; press_key('s')
                    else: release_key('s')

                    if state['index'] and not any([state['middle'], state['ring'], state['pinky']]):
                        gesture_name = "Strafe Left"; press_key('a')
                    else: release_key('a')

                    if state['index'] and state['thumb']:
                        gesture_name = "Strafe Right"; press_key('d')
                    else: release_key('d')

                    if state['thumb'] and not any([state['index'], state['middle'], state['ring'], state['pinky']]):
                        gesture_name = "Jump"; press_key(Key.space)
                    else: release_key(Key.space)

                    if state['pinky'] and not any([state['thumb'], state['index'], state['middle'], state['ring']]):
                        gesture_name = "Sneak"; press_key(Key.shift)
                    else: release_key(Key.shift)

                    if state['pinky'] and state['thumb']:
                        gesture_name = "Place / Use"; mouse.press(Button.right)
                    else: mouse.release(Button.right)

                    if all(state.values()) and abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x) > 0.5:
                        gesture_name = "Open Inventory"; press_key('e')
                    else: release_key('e')

                    if state['index'] and state['middle'] and state['ring']:
                        gesture_name = "Sprint"; press_key(Key.ctrl)
                    else: release_key(Key.ctrl)

                elif hand_label == "Left":
                    # Left hand → attack + all hotbar slots + drop item
                    if state['index'] and state['middle'] and not state['ring']:
                        gesture_name = "Attack"; mouse.press(Button.left)
                    else: mouse.release(Button.left)

                    # Hotbar 1–9
                    if state['pinky'] and not any([state['ring'], state['middle'], state['index']]):
                        gesture_name = "Hotbar 1"; press_key('1')
                    else: release_key('1')
                    if state['ring'] and not any([state['middle'], state['index']]):
                        gesture_name = "Hotbar 2"; press_key('2')
                    else: release_key('2')
                    if state['middle'] and not state['index']:
                        gesture_name = "Hotbar 3"; press_key('3')
                    else: release_key('3')
                    if state['index'] and not state['middle']:
                        gesture_name = "Hotbar 4"; press_key('4')
                    else: release_key('4')
                    if state['thumb'] and not any([state['index'], state['middle'], state['ring'], state['pinky']]):
                        gesture_name = "Hotbar 5"; press_key('5')
                    else: release_key('5')
                    if state['thumb'] and state['index']:
                        gesture_name = "Hotbar 6"; press_key('6')
                    else: release_key('6')
                    if state['thumb'] and state['middle']:
                        gesture_name = "Hotbar 7"; press_key('7')
                    else: release_key('7')
                    if state['thumb'] and state['ring']:
                        gesture_name = "Hotbar 8"; press_key('8')
                    else: release_key('8')
                    if state['thumb'] and state['pinky']:
                        gesture_name = "Hotbar 9"; press_key('9')
                    else: release_key('9')

                    # Drop item
                    if state['index'] and state['pinky']:
                        gesture_name = "Drop Item"; press_key('q')
                    else: release_key('q')

                # Stabilization
                gesture_buffers[hand_label].append(gesture_name)
                if all(g == gesture_name for g in gesture_buffers[hand_label]):
                    gestures_detected[hand_label] = gesture_name
                else:
                    gestures_detected[hand_label] = "None"

        # === Terminal output ===
        print(f"Right Hand: {gestures_detected['Right']}  |  Left Hand: {gestures_detected['Left']}")

        # === On-screen overlay ===
        y0 = 20
        for label, gesture in gestures_detected.items():
            cv2.putText(frame, f"{label}: {gesture}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y0 += 30

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
