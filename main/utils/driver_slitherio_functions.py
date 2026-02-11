from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException # marche sans ?
import cv2 # to fix
# --- CONFIG ---
SLITHER_URL = "https://slither.io/"


def is_game_over(driver) -> bool:
    try:
        el = driver.find_element(By.ID, "playh")
        return el.is_displayed()
    except NoSuchElementException:
        return False

def start_game(driver):
    #Wait for play button to be clickable (up to 10 seconds)
    play_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "btnt.nsi.sadg1"))
    )
    play_button.click()
    
    while is_game_over(driver):
        ...
        #print("wait for game to start")
    print("Game Start")

# --- START BROWSER ---
def open_slitherio():
    global driver
    try :
        driver = webdriver.Chrome()
    except Exception as e:
        return e
    driver.set_window_size(1024, 768)
    driver.get(SLITHER_URL)

    start_game(driver)
    
    return driver
