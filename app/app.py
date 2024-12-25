from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_layers, num_filters, kernel_size, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_channels
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            in_channels = num_filters
        self.fc = nn.Linear(num_filters * (224 // (2 ** num_layers)) ** 2, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Initialize the model with training parameters
model = SimpleCNN(num_layers=4, num_filters=32, kernel_size=3, input_channels=3, num_classes=2)
model.load_state_dict(torch.load('model/best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5059, 0.5021, 0.5020], std=[0.2108, 0.1982, 0.1949])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an example image is selected
    if 'example_image' in request.form:
        example_image = request.form['example_image']
        image_path = f'static/{example_image}'
        image = Image.open(image_path).convert('RGB')
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('result.html', error="No file selected.")
        image = Image.open(file.stream).convert('RGB')
    else:
        return render_template('result.html', error="No image provided.")

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = ['Arctic Fox', 'Elephant']
    prediction = class_names[predicted.item()]
    confidence = torch.softmax(output, dim=1)[0][predicted.item()] * 100

    # Render the result page
    return render_template('result.html', prediction=prediction, confidence=confidence.item())



if __name__ == '__main__':
    app.run(debug=True)
