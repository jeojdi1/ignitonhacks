from pynput.keyboard import Controller, Key
import time

keyboard = Controller()

time.sleep(3)  # Gives you time to click on Notepad or another text field
keyboard.press('w')
keyboard.release('w')
keyboard.press(Key.space)
keyboard.release(Key.space)
