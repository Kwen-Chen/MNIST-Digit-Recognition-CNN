import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Import the CNN model from main.py
from main import CNN

def load_model(model_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_digit(model, device, image_tensor):
    """Predict digit from image tensor"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
        confidence = F.softmax(output, dim=1).max().item()
        return prediction.item(), confidence

def visualize_predictions(model, device, test_loader, num_samples=10):
    """Visualize model predictions on test samples"""
    model.eval()
    
    # Get some test samples
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Create subplot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Get prediction
        pred, confidence = predict_digit(model, device, images[i])
        true_label = labels[i].item()
        
        # Plot image
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {pred} (Conf: {confidence:.2f})\nTrue: {true_label}')
        axes[i].axis('off')
        
        # Color code based on prediction accuracy
        if pred == true_label:
            axes[i].set_facecolor('lightgreen')
        else:
            axes[i].set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('output/prediction_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def test_model_accuracy(model, device, test_loader):
    """Test model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Model Accuracy on Test Set: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    # Load the best model
    model_path = 'output/mnist_cnn_best_acc_99.17.pth'
    
    try:
        model, device = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {device}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please run main.py first to train the model.")
        exit(1)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Test model accuracy
    accuracy = test_model_accuracy(model, device, test_loader)
    
    # Visualize some predictions
    print("\nGenerating prediction visualization...")
    fig = visualize_predictions(model, device, test_loader)
    print("Visualization saved to output/prediction_demo.png")
    
    # Interactive prediction demo
    print("\n" + "="*50)
    print("Interactive Prediction Demo")
    print("="*50)
    
    # Get a random test sample
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Show a few random predictions
    for i in range(5):
        idx = np.random.randint(0, len(images))
        image = images[idx]
        true_label = labels[idx].item()
        
        pred, confidence = predict_digit(model, device, image)
        
        print(f"Sample {i+1}:")
        print(f"  True digit: {true_label}")
        print(f"  Predicted: {pred}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Correct: {'✓' if pred == true_label else '✗'}")
        print() 