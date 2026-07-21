import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

# Initialize the configuration
data_path = "~/mobilenet_rock5/data/the-animalist-cat-vs-dog-classification/Cat vs Dog"
model_path = "~/mobilenet_rock5/models/classifier_head.pt"
epochs = 30
lr = 1e-3
img_size = (224, 224)

def load_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    #model.to("cpu")

    return model

class MobileNetV2Features(nn.Module):
    def __init__(self):
        super().__init__()

        # Load model
        mobilenet = load_model()

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.features(x)    # [1, 1280, 7, 7]
        x = self.avgpool(x)     # [1, 1280, 1, 1]
        x = torch.flatten(x, 1)    # [1, 1280]
        
        return x

class MobileNetV2Classifier(nn.Module):
    def __init__(self): # num_classes=2
        super().__init__()

        mobilenet = load_model()
        self.classifier = mobilenet.classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280)   # default: 1000 classes
         )
    
    def forward(self, x):
        x = self.classifier(x)
        return x

def prepare_data_loader(data_path):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],     # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(
        root=data_path+"/train/",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=data_path+"/test/",
        transform=transform
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)

    train_data_loader = DataLoader(
        train_dataset, batch_size=1,
        sampler=train_sampler
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=1,
        sampler=test_sampler
    )
    print(f"train_dataloader = {train_data_loader}")
    print(f"test_dataloader = {test_data_loader}")

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    class_names = train_dataset.classes
    print(f"train_dataset_size = {train_dataset_size}")
    print(f"test_dataset_size = {test_dataset_size}")

    # Visualize a batch of images before training the model
    X_batch, y_batch = next(iter(train_data_loader))
    X_batch_size = len(X_batch)

    print(f"X_batch.shape = {X_batch.shape}")
    print(f"y_batch = {y_batch.shape}")
    print(f"X_batch[0].shape = {X_batch[0].shape}")
    print(f"y_batch[0] = {y_batch[0]}")

    nrows, ncols = 2, 4
    plt.figure(figsize=(9, 4))
    for i in range(min(nrows * ncols, len(X_batch))):
        img = convert_tensor_to_image(X_batch[i])
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{class_names[y_batch[i]]}")
    plt.tight_layout()
    plt.show()

    return train_data_loader, test_data_loader, train_dataset_size, test_dataset_size

def convert_tensor_to_image(input):
    """
    Receive an example from the dataset in the tensor format, 
    turn the tensor into NumPy in the (num_channels, width, height) format.
    """
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    output = std * input + mean
    output = np.clip(output, 0, 1)

    return output

def train_model(model, model_path, criterion, optimizer, scheduler, epochs, device):
    model.to(device)
    train_data_loader, test_data_loader, train_dataset_size, test_dataset_size = prepare_data_loader(data_path)

    best_accuracy = 0.0
    #EPOCHS = 30

    since = time.time()

    for epoch in range(epochs):
        print(f"Epoch [{epoch}/{epochs - 1}]")

        # Training phase:
        model.train()
        running_loss, running_corrects = 0.0, 0.0
        for X_batch, y_batch in train_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()
                
            # Statistics
            running_loss += loss.item() * X_batch.shape[0]
            running_corrects += torch.sum(preds == y_batch.data)

        epoch_loss = running_loss / train_dataset_size
        epoch_accuracy = running_corrects.double() / train_dataset_size
        print(f"Train loss: {epoch_loss:6.5f}, Accuracy: {epoch_accuracy:5.4f}")

        scheduler.step()     # update the learning rate

        # Evaluation phase:
        model.eval()
        running_loss, running_corrects = 0.0, 0.0

        for X_batch, y_batch in test_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.set_grad_enabled(False):
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                loss = criterion(logits, y_batch)
            
            running_loss += loss.item() * X_batch.shape[0]
            running_corrects += torch.sum(preds == y_batch.data) 

        epoch_loss = running_loss / test_dataset_size
        epoch_accuracy = running_corrects.double() / test_dataset_size
        print(f"Test loss: {epoch_loss:6.5f}, Accuracy: {epoch_accuracy:5.4f}")
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            # Save the model
            torch.save(model.state_dict(), model_path)

    telapsed = time.time() - since
    print(f"Training complete in {telapsed // 60:.0f} minutes {telapsed % 60:.0f} seconds")
    print(f"Best validiation accuracy: {best_accuracy}")

    return model

def export_to_onnx():
    model = MobileNetV2Classifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Extract just the feature extractor part
    model_features = MobileNetV2Features()
    #model_features.load_state_dict(model_features.state_dict())
    model_features.eval()

    batch_size = 8
    sample_inputs = torch.randn(batch_size, 3, 224, 224)
    output = model(sample_inputs)
    print("Output shape:", output.shape)

    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / "mobilenetv2_features.onnx"

    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_inputs,
            onnx_path,
            opset_version = 19,
            input_names = ["input"],
            output_names = ["features"],
            #dynamo = True,
            dynamic_shapes = None   # Fixed the model to same shapes
        )

    print("-I- Model file is exported here: ", onnx_path.resolve())


if __name__ == "__main__":
    # Select device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu") # using CPU only
    print(f"Device: {device}")

    model = MobileNetV2Classifier().to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) #optim.SGD(model_A.parameters(), lr=0.001, momentum=0.9)      
    criterion = nn.CrossEntropyLoss()
    # The learning rate will be reduced by multiplier of 0.1 after every 7 epochs.
    scheduler  = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(model, model_path, criterion, optimizer, scheduler, epochs=epochs, device=device)

    export_to_onnx()

   

