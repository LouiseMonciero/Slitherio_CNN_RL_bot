# Slitherio Bot 🐍
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)

A Slither.io bot that plays autonomously using computer vision ✨

## Features

- 🎮 **Autonomous gameplay** - AI bot plays Slither.io without human intervention.
- 📊 **Real-time score detection** - Uses CNN to read the score from the game
- 🔄 **Auto-restart** - Automatically restarts the game when game over is detected
- 📸 **Data collection** - Can record screenshots for you to have training data and adapt the models as you wish to.
- 🖥️ **Preview mode** - Optional real-time preview of the bot's view

All the listed features will be implemented soon. For now on, the autonomous gameplay is not available.

## Project structure

```
Slitherio_CNN_RL_bot/
│
├── main/
│   ├── main.py                 # Main entry point
│   ├── utils/
│   │   ├── driver_slitherio_functions.py  # Selenium browser control
│   │   └── predictions_score.py           # Score prediction (OCR)
│   │
│   └── model/
│       ├── cnn.ipynb                  # CNN model training notebook
│       └── cnn.pth                    # Trained CNN weights
│
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- Google Chrome browser
- Pytorch

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Slitherio_CNN_RL_bot.git
cd Slitherio_CNN_RL_bot
```

2. **Create and activate virtual environment**
```bash
cd main
python3 -m venv .venv
source ./.venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r ../requirements.txt
```

4. **Connect to environement**
```bash
source .venv/bin/activate
```

## Usage

### Run the bot

```bash
python main.py
```

### Controls

- Press **ESC** to stop the bot

## Requirements

```
selenium
opencv-python
mss
pynput
numpy
torch
torchvision
pandas
```

## Notes

⚠️ **Important**: Do not cover the browser window while the bot is running, otherwise the bot will be unable to see the game and screen capture won't work properly for data collection.

## License

MIT License