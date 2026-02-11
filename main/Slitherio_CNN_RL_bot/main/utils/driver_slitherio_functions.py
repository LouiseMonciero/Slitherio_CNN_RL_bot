from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import cv2
import matplotlib.pyplot as plt
import tkinter as tk

# --- CONFIG ---
SLITHER_URL = "https://slither.io/"

def is_game_over(driver) -> bool:
    try:
        el = driver.find_element(By.ID, "playh")
        return el.is_displayed()
    except NoSuchElementException:
        return False

def start_game(driver):
    play_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "btnt.nsi.sadg1"))
    )
    play_button.click()
    
    while is_game_over(driver):
        print("wait for game to start")

def plot(scores, mean_scores, train=False):
    if train:
        root = tk.Tk()
        root.title("Training Progress")

        fig = plt.figure()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.ylim(ymin=0)

        canvas = plt.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def update_plot():
            plt.clf()
            plt.plot(scores, label='Score')
            plt.plot(mean_scores, label='Mean Score')
            plt.legend()
            plt.draw()
            root.after(100, update_plot)

        update_plot()
        tk.Button(root, text="Quit", command=root.quit).pack(side=tk.BOTTOM)
        root.mainloop()
    else:
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

# --- START BROWSER ---
def open_slitherio():
    global driver
    try:
        driver = webdriver.Chrome()
    except Exception as e:
        return e
    driver.set_window_size(1024, 768)
    driver.get(SLITHER_URL)

    start_game(driver)
    
    return driver