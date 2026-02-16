import cv2
import math
import tkinter as tk
from tkinter import Tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.predictions_score import predict_score


class Game_slitherio:

    def __init__(self, center, mouse_position):
        self.is_game_over = True
        self.score = 0
        self.direction = 90 # angle from sinus axis in trigo circle with center as the head of the snake.
        self.absolute_center_frame = center # a tuple with absolute px pos in the screen.
        self.frame = None
        self.preprocessed_me_frame = None
        self.preprocessed_dots_frame = None
        self.preprocessed_snake_frame = None


    def clalculate_reward(self):
        return 0
    
    def reset(self, mouse_position):
        # init game state
        self.is_game_over = True
        self.score = 0
        self.direction =  round(math.degrees(math.atan2(mouse_position[1] - self.absolute_center_frame[1],
                                        mouse_position[0] - self.absolute_center_frame[0])))
        
        

        self.frame = None
        self.preprocessed_me_frame = None
        self.preprocessed_dots_frame = None
        self.preprocessed_snake_frame = None

        
    def render(self, frame, mouse_position):

        def get_chanels_from_frame(frame):
            
            def fill_closed_regions(edge_img, close_ksize=5, close_iter=2, min_area=30):
                """
                edge_img : image BGR ou gray avec contours blancs sur fond noir
                Retour : mask rempli (uint8 0/255)
                """

                # 1) Gray
                if len(edge_img.shape) == 3:
                    gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = edge_img.copy()

                # 2) Binariser (contours -> 255)
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 3) Fermer les petits trous/ruptures pour "boucler" les contours
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
                bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iter)

                # Optionnel: épaissir un peu les traits pour mieux fermer
                bw_closed = cv2.dilate(bw_closed, k, iterations=1)

                # 4) Contours externes + remplissage
                contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                filled = np.zeros_like(gray, dtype=np.uint8)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_area:
                        cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

                return filled

            def seg_3_chanels(filled_img, max_thickness, min_width=10):
                """From edge image, 
                Chanel 1 : dots
                Chanel 2 : snakes
                Chanel 3 : snake of the user"""
                # Trouver les forme dans l'image
                forme, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out_dot = np.zeros_like(filled_img)
                out_me = np.zeros_like(filled_img)
                out_snakes = np.zeros_like(filled_img)
                H, W = filled_img.shape[:2]
                
                center = (W // 2, H // 2)

                for cnt in forme:
                    # Calculer l'aire et le périmètre de la forme
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = (4 * np.pi * area) / (perimeter ** 2)

                    # Calculer l'épaisseur moyenne de la forme
                    thickness = perimeter / (2 * np.sqrt(np.pi * area))
                    x, y, w, h = cv2.boundingRect(cnt)
                    fill_ratio = area / (w * h + 1e-6)
                    min_dim = min(w, h)  # Largeur minimale (plus petite dimension)


                    if cv2.pointPolygonTest(cnt, center, False) >= 0:
                        cv2.drawContours(out_me, [cnt], -1, 255, thickness=cv2.FILLED)
                    
                    elif (circularity > 0.55):
                        cv2.drawContours(out_dot, [cnt], -1, 255, thickness=cv2.FILLED)

                    else:
                        if (min_dim > 20):
                            cv2.drawContours(out_snakes, [cnt], -1, 255, thickness=cv2.FILLED)
                return out_dot, out_me, out_snakes

            # retirer le bandeau 
            frame = frame[150: , :]
            edges = cv2.Canny(frame, 100, 200)
            filled = fill_closed_regions(edges)
            return seg_3_chanels(filled, 2)

        self.frame = frame
        #temp = predict_score(self.frame)
        temp = predict_score(self.frame, model=True)
        self.score = temp if temp is not None else self.score # keep the old score in order to prevent issues if the prediction was Null

        
        # angle
        self.direction =  round(math.degrees(math.atan2(mouse_position[1] - self.absolute_center_frame[1],
                                        mouse_position[0] - self.absolute_center_frame[0])))
        
        
        self.preprocessed_me_frame, self.preprocessed_dots_frame, self.preprocessed_snake_frame = get_chanels_from_frame(frame)


class training_Window:
    def __init__(self, game):
        self.game = game

        self.root_score_tk = tk.Tk()
        self.root_score_tk.title("Training of the agent")

        # taille + position bas droite
        w, h = 550, 350
        sw = self.root_score_tk.winfo_screenwidth()
        sh = self.root_score_tk.winfo_screenheight()
        self.root_score_tk.geometry(f"{w}x{h}+{sw-w-10}+{sh-h-60}")
        self.root_score_tk.resizable(False, False)

        self.scores = []
        self.mean_scores = []

        # Configuration matplotlib avec polices réduites
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 9,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
        })

        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.2), dpi=100)
        self.fig.tight_layout(pad=2.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root_score_tk)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        self._closed = False
        self.root_score_tk.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self._closed = True
        plt.close(self.fig)
        self.root_score_tk.destroy()

    def update_plot(self, score):
        if self._closed:
            return
            
        self.scores.append(score)
        mean_score = sum(self.scores) / len(self.scores)
        self.mean_scores.append(mean_score)

        self.ax.clear()
        
        x = list(range(len(self.scores)))
        
        # Scatter plot avec points reliés pour les scores
        self.ax.plot(x, self.scores, 'b-', linewidth=1, alpha=0.7)
        self.ax.scatter(x, self.scores, c='blue', s=15, zorder=5, label='Score')
        
        # Ligne moyenne
        self.ax.plot(x, self.mean_scores, 'r-', linewidth=1.5, label='Moyenne')
        
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Épisode")
        self.ax.set_ylabel("Score")
        self.ax.set_ylim(bottom=0)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def step(self):
        """À appeler régulièrement dans ta boucle principale."""
        if self._closed:
            return
        try:
            self.root_score_tk.update_idletasks()
            self.root_score_tk.update()
        except tk.TclError:
            self._closed = True