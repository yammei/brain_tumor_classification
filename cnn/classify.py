import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from cnn_model import CNNModel

# labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# load model
model = CNNModel()
checkpoint = torch.load('cnn_brain_tumor_classification_v1.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # batch size
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        # confidence score
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()

    print(f"prediction: {classes[class_idx]} (confidence: {confidence.item() * 100:.2f}%)")
    print("class probabilities:")
    for idx, prob in enumerate(probabilities[0]):
        print(f"  {classes[idx]}: {prob.item() * 100:.2f}%")

    return classes[class_idx], probabilities[0].tolist()

if __name__ == "__main__":
    input_to_classification_map = {
        '1': 'glioma',
        '2': 'meningioma',
        '3': 'pituitary',
        '4': 'notumor'
    }
    prompt = f"please enter which type to classify:\n1 - glioma\n2 - mengioma\n3 - pituitary\n4 - no tumor\n"
    img_path = f"showcase_images/test_image_{input_to_classification_map[str(input(prompt))]}.jpg"
    predict_image(img_path)
