import torch
from model import CNN
from torchvision import transforms
import numpy as np

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt', map_location=device))
    model.eval()
    return model, device

def predict_image(image, model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item() 