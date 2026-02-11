# Slitherio CNN RL Bot

This project implements a reinforcement learning bot for the popular online game Slither.io using a Convolutional Neural Network (CNN). The bot interacts with the game through a Selenium WebDriver, capturing frames and processing them to make decisions based on the game state.

## Project Structure

```
Slitherio_CNN_RL_bot
├── main
│   ├── main.py                     # Main entry point of the application
│   ├── utils
│   │   ├── enivronement.py         # Manages game state and frame processing
│   │   ├── driver_slitherio_functions.py # Interacts with the Slither.io game using Selenium
│   │   ├── predictions_score.py     # Predicts scores based on game frames
│   │   └── train.py                 # Responsible for training the model
│   └── data
│       └── screenshots              # Directory for storing screenshots
└── README.md
```

## Requirements

- Python 3.x
- Selenium
- OpenCV
- NumPy
- Matplotlib (for plotting scores)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Slitherio_CNN_RL_bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the Chrome WebDriver installed and available in your PATH.

## Usage

1. Open the terminal and navigate to the project directory.
2. Run the main script:
   ```
   python main/main.py
   ```

3. The bot will automatically open the Slither.io game in a browser window and start playing.

## Training

To train the model, modify the `train` parameter in `main.py` to `True`. This will enable the training mode, allowing the bot to learn from its gameplay.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.