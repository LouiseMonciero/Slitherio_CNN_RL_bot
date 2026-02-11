import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TrainingPlotter:
    def __init__(self, master):
        self.master = master
        self.master.title("Training Progress")
        
        self.scores = []
        self.mean_scores = []
        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_title('Training...')
        self.ax.set_xlabel('Number of Games')
        self.ax.set_ylabel('Score')
        self.ax.set_ylim(ymin=0)

    def update_plot(self, scores, mean_scores):
        self.scores = scores
        self.mean_scores = mean_scores
        
        self.ax.clear()
        self.ax.set_title('Training...')
        self.ax.set_xlabel('Number of Games')
        self.ax.set_ylabel('Score')
        self.ax.set_ylim(ymin=0)
        
        self.ax.plot(self.scores, label='Score')
        self.ax.plot(self.mean_scores, label='Mean Score')
        self.ax.text(len(self.scores)-1, self.scores[-1], str(self.scores[-1]))
        self.ax.text(len(self.mean_scores)-1, self.mean_scores[-1], str(self.mean_scores[-1]))
        self.ax.legend()
        
        self.canvas.draw()

def plot(scores, mean_scores):
    if train:  # Assuming 'train' is a global variable indicating training mode
        root = tk.Tk()
        plotter = TrainingPlotter(root)
        
        plotter.update_plot(scores, mean_scores)
        
        def update():
            plotter.update_plot(scores, mean_scores)
            root.after(1000, update)  # Update every second
        
        update()
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