import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytesseract

CNN_PATH = './model/cnn.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5) # I only have 1 channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 1, 80)
        self.fc2 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 10) # predict 10 classes for the 10 digits.

    def forward(self, x):
        # -> n_batch, 1, 20, 14
        x = self.pool(F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                       # -> n_batch, 10
        return x



def leave_only_white(img):
    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = (0, 0, 200)
    upper_white = (180, 40, 255)
    
    mask = cv2.inRange(hsv, lower_white, upper_white)

    result = img.copy()
    result[mask == 0] = [0, 0, 0]
    return result

def preprocess_image(image):
    """
    Preprocess the image to optimize the predictions of the CNN model by :
        - only keep the pixel of the score as visible
        - applying grayscale
        - croping on only the number of the score in the window
    """

    # croping
    h, _ = image.shape[:2]
    image = image[(h - 39): (h - 26), 107:210]

    image = leave_only_white(image)

    # applying grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray reduce the image to 1 channel.
    cv2.imwrite('./temp.png', image)

    return image


def predict_score(image, model=False): # As the CNN model doesn't work perfclty, it is not wet use to train the CNN RL
    
    image = preprocess_image(image)
    score = 0
    
    if model is False :
        try :
            res = int(pytesseract.image_to_string(image, config='--psm 6 digits'))
        except Exception:
            return None

    # load the CNN model
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(CNN_PATH, weights_only=True))
    model.eval() # deactivate Dropout...
    
    for i in range(8): # 8 digits of the score
        crop_left = (8*i)-1 if i!= 0 else 0
        img_digit = image[:, crop_left:(8 * (i+1))]

        if np.all(img_digit == 0):
            continue
        
        img_digit = cv2.resize(img_digit, (20, 14))
        x = torch.from_numpy(img_digit).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(DEVICE)
        with torch.no_grad():
            digit = torch.argmax(model(x), dim=1).item()
            digit = 0 if digit==10 else digit
        
        score = score * 10 + digit
    
    return score

""" img = cv2.imread("./main/data/train/screenshots/screenshot4.jpg")
print(predict_score(img)) """