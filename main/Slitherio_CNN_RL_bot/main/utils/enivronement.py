import cv2
import math
import numpy as np
from utils.predictions_score import predict_score
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Game_slitherio:

    def __init__(self, center):
        self.reset()
        self.absolute_center_frame = center
        self.frame = None
        self.preprocessed_me_frame = None
        self.preprocessed_dots_frame = None
        self.preprocessed_snake_frame = None

    def calculate_reward(self):
        return 0
    
    def reset(self):
        self.is_game_over = True
        self.score = 0
        self.direction = 90

        self.frame = None
        self.preprocessed_me_frame = None
        self.preprocessed_dots_frame = None
        self.preprocessed_snake_frame = None

    def get_channels_from_frame(self, frame):
        
        def fill_closed_regions(edge_img, close_ksize=5, close_iter=2, min_area=30):
            if len(edge_img.shape) == 3:
                gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = edge_img.copy()

            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iter)
            bw_closed = cv2.dilate(bw_closed, k, iterations=1)

            contours, _ = cv2.findContours(bw_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(gray, dtype=np.uint8)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

            return filled

        def seg_3_channels(filled_img, max_thickness, min_width=10):
            forme, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_dot = np.zeros_like(filled_img)
            out_me = np.zeros_like(filled_img)
            out_snakes = np.zeros_like(filled_img)
            H, W = filled_img.shape[:2]
            center = (W // 2, H // 2)

            for cnt in forme:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                thickness = perimeter / (2 * np.sqrt(np.pi * area))
                x, y, w, h = cv2.boundingRect(cnt)
                min_dim = min(w, h)

                if cv2.pointPolygonTest(cnt, center, False) >= 0:
                    cv2.drawContours(out_me, [cnt], -1, 255, thickness=cv2.FILLED)
                elif circularity > 0.55:
                    cv2.drawContours(out_dot, [cnt], -1, 255, thickness=cv2.FILLED)
                else:
                    if min_dim > 20:
                        cv2.drawContours(out_snakes, [cnt], -1, 255, thickness=cv2.FILLED)
            return out_dot, out_me, out_snakes

        frame = frame[150:, :]
        edges = cv2.Canny(frame, 100, 200)
        filled = fill_closed_regions(edges)
        return seg_3_channels(filled, 2)

    def render(self, frame, mouse_position):
        self.frame = frame
        self.score = predict_score(self.frame)
        
        self.direction = round(math.degrees(math.atan2(mouse_position[1] - self.absolute_center_frame[1],
                                      mouse_position[0] - self.absolute_center_frame[0])))
        
        self.preprocessed_me_frame, self.preprocessed_dots_frame, self.preprocessed_snake_frame = self.get_channels_from_frame(frame)

    def plot(self, scores, mean_scores):
        root = tk.Tk()
        root.title("Training Scores")

        fig, ax = plt.subplots()
        ax.set_title('Training...')
        ax.set_xlabel('Number of Games')
        ax.set_ylabel('Score')
        ax.plot(scores, label='Scores')
        ax.plot(mean_scores, label='Mean Scores')
        ax.set_ylim(ymin=0)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.Button(root, text="Quit", command=root.quit).pack(side=tk.BOTTOM)

        tk.mainloop()