import time
import cv2
import numpy as np
import os
import mss
from pynput import keyboard
from pynput.mouse import Controller

from utils.driver_slitherio_functions import open_slitherio, start_game, is_game_over
from utils.enivronement import Game_slitherio, training_Window
from utils.agent import Agent

def record_selenium_window(
        driver, 
        fps=20, 
        preview:bool=True,
        collecting:bool=False, 
        images_dir='./data/screenshots/',
        restart:bool=True,
        bot_playing:bool=True,
        train:bool=True,
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

    # Le bandeau du moniteur est de 15Opx environ.
    center = (
        (size["height"] // 2) +pos["x"] + 150,
        (size["width"] // 2) + pos['y']
    ) # (x, y)
    
    mouse = Controller()
    mouse.position = (center[0] + 100, center[1])

    os.makedirs(images_dir, exist_ok=True)
    nb_img = 26


    stop = False
    ep = 0 #episode / render loop number
    env = Game_slitherio(center, mouse.position) # init

    # Initialiser l'agent
    agent = None
    prev_state = None
    if bot_playing: # harmonise plus tard avec train=True
        agent = Agent()
        # agent.load("model.pth")  # Décommenter pour charger un modèle existant


    if train:
        training_window = training_Window(env)
        training_window.step()

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
                    print(" -------- end of game ------")

                    if train and prev_state is not None:
                        # Stocker la transition finale avec pénalité
                        reward = agent.calculate_reward(env, is_game_over=True)
                        current_state = agent.preprocess_state(env)
                        agent.store_transition(prev_state, prev_action_idx, reward, current_state, done=True)
                        agent.decay_epsilon()
                    
                    env.reset(mouse.position)
                    if agent:
                        agent.reset_episode()
                    mouse.position = (center[0] + 100, center[1])
                    prev_state = None

                    env.reset(mouse.position) #score = 0
                    mouse.position = (center[0] + 100, center[1])

                    if restart:
                        print(' ---- starting... --- ')
                        time.sleep(0.5) # wait for the button to open.
                        start_game(driver)
                

                # Predict score with CNN and OpenCV preprocessing
                #score = (predict_score(frame))
                env.render(frame, mouse.position)
                print(f"Ep n°{ep} | score {env.score} | direction {env.direction} .") 
                if agent:
                    current_state = agent.preprocess_state(env)
                    
                    # Stocker la transition précédente
                    if train and prev_state is not None:
                        reward = agent.calculate_reward(env, is_game_over=False)
                        agent.store_transition(prev_state, prev_action_idx, reward, current_state, done=False)
                        
                        # Entraîner
                        loss = agent.train_step()
                    
                    # Sélectionner une action
                    action_idx, angle = agent.select_action(env)
                    prev_action_idx = action_idx
                    prev_state = current_state
                    
                    # Déplacer la souris selon l'action
                    distance = 100
                    new_x = center[0] + distance * np.cos(np.radians(angle))
                    new_y = center[1] + distance * np.sin(np.radians(angle))
                    mouse.position = (int(new_x), int(new_y))
                        
                    
                if train:
                        training_window.step()
                        if ep % 5 == 0:
                            training_window.update_plot(env.score) 
                # the score and frame will be then used to guide the IA bot with Reinforcement Learning.

            dt = time.time() - t0
            if dt < frame_dt:
                time.sleep(frame_dt - dt)
            ep+=1

    if preview:
        cv2.destroyAllWindows()
    if train:
        training_window.close()
    if agent:
            agent.save("RL_model.pth")


if __name__ == "__main__":
    driver = open_slitherio()

    driver.set_window_size(1024, 768)


    # MAKE SURE you don't cover with anything the window of the slitherio webdriver, otherwise, the recording won't work.
    record_selenium_window(
        driver, 
        images_dir='./data/screenshots_more/',
        fps=2, 
        preview=False, 
        collecting=True,
        bot_playing=False, 
        train=False)
    

    driver.quit()