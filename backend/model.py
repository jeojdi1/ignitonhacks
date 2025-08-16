import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
from collections import deque
import math
import socket
import json

keyboard = KeyboardController()
mouse = MouseController()

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh  # <--- restore face mesh
mp_drawing = mp.solutions.drawing_utils

HOST = '127.0.0.1'
PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
except ConnectionRefusedError:
    print(f"Warning: Could not connect to {HOST}:{PORT}. Data will not be sent.")
    sock = None

pressed_keys = set()
pressed_mouse = set()

def finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def thumb_up(hand_landmarks, hand_label="Right"):
    tip = hand_landmarks.landmark[4]
    mcp = hand_landmarks.landmark[2]
    if hand_label == "Right":
        return tip.x < mcp.x
    else:
        return tip.x > mcp.x

def get_finger_states(hand_landmarks, hand_label="Right"):
    return {
        'thumb':  thumb_up(hand_landmarks, hand_label),
        'index':  finger_up(hand_landmarks, 8, 6),
        'middle': finger_up(hand_landmarks, 12, 10),
        'ring':   finger_up(hand_landmarks, 16, 14),
        'pinky':  finger_up(hand_landmarks, 20, 18)
    }

def press_key(k): 
    try:
        keyboard.press(k)
        pressed_keys.add(str(k))
    except: pass

def release_key(k):
    try:
        keyboard.release(k)
        pressed_keys.discard(str(k))
    except: pass

def press_mouse(btn):
    try:
        mouse.press(btn)
        pressed_mouse.add(str(btn))
    except: pass

def release_mouse(btn):
    try:
        mouse.release(btn)
        pressed_mouse.discard(str(btn))
    except: pass

buffer_length = 3
gesture_buffers = {'Left': deque(maxlen=buffer_length), 'Right': deque(maxlen=buffer_length)}

cap = cv2.VideoCapture(0)
window_name = "Minecraft Gesture + Head Controller"
angle_sensitivity = 10  # Increased sensitivity for faster movement

neutral_yaw = 0
neutral_pitch = 0
calibrated = False

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:  # <--- restore face mesh context
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            calibrated = False

        gestures_detected = {'Left': "None", 'Right': "None"}
        results_hands = hands.process(rgb)

        yaw_deg = pitch_deg = 0

        # --- Head pointer movement ---
        results_face = face_mesh.process(rgb)
        pointer_x = pointer_y = None
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            nose = face_landmarks[1]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            mouth_top = face_landmarks[13]
            mouth_bottom = face_landmarks[14]

            if not calibrated:
                neutral_yaw = nose.x - (left_eye.x + right_eye.x)/2
                neutral_pitch = nose.y - ((mouth_top.y + mouth_bottom.y)/2)
                calibrated = True
                print("Neutral head position calibrated!")

            # Yaw: horizontal movement (nose vs eyes center)
            yaw = (nose.x - (left_eye.x + right_eye.x)/2) - neutral_yaw
            # Pitch: vertical movement (nose vs mouth center)
            pitch = (nose.y - ((mouth_top.y + mouth_bottom.y)/2)) - neutral_pitch

            yaw_deg = yaw * angle_sensitivity * 100
            pitch_deg = -pitch * angle_sensitivity * 100  # Invert pitch so up is up

            mouse.move(int(yaw_deg), int(pitch_deg))

            pointer_x = int(w // 2 + yaw_deg)
            pointer_y = int(h // 2 + pitch_deg)
            cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 0, 255), -1)

        # --- Finger gestures ---
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                state = get_finger_states(hand_landmarks, hand_label)
                gesture_name = "None"

                if hand_label == "Right":
                    # Jump: thumb up only
                    if state['thumb'] and not any([state['index'], state['middle'], state['ring'], state['pinky']]):
                        gesture_name = "Jump"
                        press_key(Key.space)
                    else:
                        release_key(Key.space)

                    # Sneak: pinky only
                    if state['pinky'] and not any([state['thumb'], state['index'], state['middle'], state['ring']]):
                        gesture_name = "Sneak"
                        press_key(Key.shift)
                    else:
                        release_key(Key.shift)

                    # Place / Use: pinky + thumb
                    if state['pinky'] and state['thumb']:
                        gesture_name = "Place / Use"
                        press_mouse(Button.right)
                    else:
                        release_mouse(Button.right)

                    # Open Inventory: all fingers up but thumb/pinky far apart
                    if all(state.values()) and abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x) > 0.5:
                        gesture_name = "Open Inventory"
                        press_key('e')
                    else:
                        release_key('e')

                    # Sprint: index + middle + ring
                    if state['index'] and state['middle'] and state['ring']:
                        gesture_name = "Sprint"
                        press_key(Key.ctrl)
                    else:
                        release_key(Key.ctrl)

                elif hand_label == "Left":
                    # Attack / hotbar / drop gestures
                    if state['index'] and state['middle'] and not state['ring']:
                        gesture_name = "Attack"
                        press_mouse(Button.left)
                    else:
                        release_mouse(Button.left)

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

                gesture_buffers[hand_label].append(gesture_name)
                if all(g == gesture_name for g in gesture_buffers[hand_label]):
                    gestures_detected[hand_label] = gesture_name
                else:
                    gestures_detected[hand_label] = "None"

        print(f"Right Hand: {gestures_detected['Right']}  |  Left Hand: {gestures_detected['Left']}  | Yaw={yaw_deg:.1f}°, Pitch={pitch_deg:.1f}°")

        data = {
            "right_hand": gestures_detected['Right'],
            "left_hand": gestures_detected['Left'],
            "yaw": yaw_deg,
            "pitch": pitch_deg,
            "pressed_keys": list(pressed_keys),
            "pressed_mouse": list(pressed_mouse)
        }
        if sock:
            try:
                sock.sendall((json.dumps(data) + "\n").encode())
                print("Sent:", data)
            except Exception as e:
                print("Socket send error:", e)

        y0 = 30
        for label, gesture in gestures_detected.items():
            cv2.putText(frame, f"{label}: {gesture}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y0 += 30
        cv2.putText(frame, f"Head Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
