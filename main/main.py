import time
import cv2
import numpy as np
import os
import mss
from pynput import keyboard

from utils.driver_slitherio_functions import open_slitherio, start_game, is_game_over
from utils.predictions_score import predict_score

def record_selenium_window(
        driver, 
        output_path="slither_window.mp4", 
        fps=20, 
        preview:bool=True, 
        collecting:bool=False, 
        images_dir='./data/screenshots/',
        restart:bool=True,
        bot_playing:bool=True,
        ):
    """ Params:
        - preview : if true, will show the recording in reel time
        - collecting:bool : if true, will save the images in a ./data folder
        - restart_game:bool : if true, will restart the game whenever the game is over.
    """

    pos = driver.get_window_position()   # {'x': ..., 'y': ...}
    size = driver.get_window_size()      # {'width': ..., 'height': ...}

    monitor = {
        "left": int(pos["x"]),
        "top": int(pos["y"]),
        "width": int(size["width"]),
        "height": int(size["height"]),
    }

    os.makedirs(images_dir, exist_ok=True)
    nb_img = 26


    stop = False

    def on_press(key):
        nonlocal stop
        if key == keyboard.Key.esc:
            stop = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    if preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    frame_dt = 1.0 / fps
    print("Recording window... Press ESC to stop.")

    with mss.mss() as sct:
        while not stop:
            t0 = time.time()

            img = np.array(sct.grab(monitor))               # BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)   # BGR

            if preview:
                cv2.imshow("preview", frame)
                cv2.waitKey(1)  # juste pour rafraîchir la fenêtre
            
            if collecting:
                # save image
                image_path = os.path.join(
                    images_dir, f"screenshot{nb_img}.jpg"
                )
                nb_img+=1
                cv2.imwrite(image_path, frame)

            if bot_playing: 
                if is_game_over(driver):
                    score = 0
                    if restart:
                        time.sleep(0.5) # wait for the button to open.
                        start_game(driver)
                
                # Predict score with CNN and OpenCV preprocessing
                score = (predict_score(frame))
                # the score and frame will be then used to guide the IA bot with Reinforcement Learning.

            dt = time.time() - t0
            if dt < frame_dt:
                time.sleep(frame_dt - dt)

    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    driver = open_slitherio()

    driver.set_window_size(1024, 768)


    # MAKE SURE you don't cover with anything the window of the slitherio webdriver, otherwise, the recording won't work.
    record_selenium_window(driver, output_path="slither_window.mp4", fps=2, preview=False, collecting=False)
    

    driver.quit()