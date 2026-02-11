import time
import cv2
import numpy as np
import os
import mss
from pynput import keyboard
from pynput.mouse import Controller
import tkinter as tk
from tkinter import ttk
from utils.driver_slitherio_functions import open_slitherio, start_game, is_game_over
from utils.predictions_score import predict_score
from utils.enivronement import Game_slitherio
from utils.train import display_avg_score

def record_selenium_window(
        driver, 
        output_path="slither_window.mp4", 
        fps=20, 
        preview:bool=True, 
        collecting:bool=False, 
        images_dir='./data/screenshots/',
        restart:bool=True,
        bot_playing:bool=True,
        train:bool=True,
        ):
    pos = driver.get_window_position()
    size = driver.get_window_size()

    monitor = {
        "left": int(pos["x"]),
        "top": int(pos["y"]),
        "width": int(size["width"]),
        "height": int(size["height"]),
    }

    center = (
        (size["height"] // 2) + pos["x"] + 150,
        (size["width"] // 2) + pos['y']
    )

    mouse = Controller()
    mouse.position = (center[0] + 100, center[1])

    os.makedirs(images_dir, exist_ok=True)
    nb_img = 26

    stop = False
    env = Game_slitherio(center)

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

    if train:
        root = tk.Tk()
        root.title("Training Progress")
        scores_frame = ttk.Frame(root)
        scores_frame.pack(padx=10, pady=10)

        scores_label = ttk.Label(scores_frame, text="Scores:")
        scores_label.pack()

        scores_listbox = tk.Listbox(scores_frame, width=50, height=10)
        scores_listbox.pack()

        mean_scores_label = ttk.Label(scores_frame, text="Mean Scores:")
        mean_scores_label.pack()

        mean_scores_listbox = tk.Listbox(scores_frame, width=50, height=10)
        mean_scores_listbox.pack()

        def update_plot(scores, mean_scores):
            scores_listbox.delete(0, tk.END)
            for score in scores:
                scores_listbox.insert(tk.END, score)

            mean_scores_listbox.delete(0, tk.END)
            for mean_score in mean_scores:
                mean_scores_listbox.insert(tk.END, mean_score)

        root.after(100, lambda: update_plot([], []))  # Initial call with empty data

    with mss.mss() as sct:
        while not stop:
            t0 = time.time()

            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if preview:
                cv2.imshow("preview", frame)
                cv2.waitKey(1)

            if collecting:
                image_path = os.path.join(images_dir, f"screenshot{nb_img}.jpg")
                nb_img += 1
                cv2.imwrite(image_path, frame)

            if bot_playing:
                if is_game_over(driver):
                    env.reset()
                    mouse.position = (center[0] + 100, center[1])

                    if restart:
                        time.sleep(0.5)
                        start_game(driver)

                env.render(frame, mouse.position)

            dt = time.time() - t0
            if dt < frame_dt:
                time.sleep(frame_dt - dt)

    if preview:
        cv2.destroyAllWindows()

    if train:
        root.mainloop()

if __name__ == "__main__":
    driver = open_slitherio()
    driver.set_window_size(1024, 768)
    record_selenium_window(driver, output_path="slither_window.mp4", fps=2, preview=False, collecting=False, bot_playing=True, train=True)
    driver.quit()