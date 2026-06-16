import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Model configuration
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the Unet model
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=1,  # Input channels for grayscale images
    classes=1,      # Output classes
    activation=None
)

# Load the saved model weights
model.load_state_dict(torch.load('best_model.pth',map_location='cpu'))
model.to(DEVICE)
model.eval()  # Set the model to evaluation mode

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (224, 224))  # Resize to match the model input size
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Add channel dimension (1 channel for grayscale)
    return torch.tensor(image).to(DEVICE)

# Function to make a prediction
def predict(image_path):
    input_image = preprocess_image(image_path)
    
    with torch.no_grad():  # Disable gradient calculation
        output_logits = model(input_image)
        
    # Apply sigmoid to get probabilities and threshold
    output_prob = torch.sigmoid(output_logits)
    prediction = (output_prob > 0.5).float()  # Convert to binary mask
    
    return prediction.squeeze().cpu().numpy()  # Move back to CPU and remove unnecessary dimensions

# Path to the image you want to predict
image_path = r'data\train\tumor\163_jpg.rf.457f1210d7e0ef0332cf4970510d540e.jpg'  # Change to the path of your image

# Make a prediction
predicted_mask = predict(image_path)

# Visualize the input image and predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()