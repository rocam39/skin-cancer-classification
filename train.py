import os
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "data/HAM10000_metadata.csv"
IMAGE_DIR = "data/images"

MODEL_PATH = "models/best_skin_cancer_resnet50_model.pth"
ENCODER_PATH = "models/label_encoder.pkl"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

NUM_CLASSES = 7
RANDOM_STATE = 42


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(CSV_PATH)

print(metadata.head())
print("\nClass distribution:")
print(metadata["dx"].value_counts())


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()
metadata["encoded_labels"] = label_encoder.fit_transform(metadata["dx"])

print("\nClasses:")
print(label_encoder.classes_)


# ============================================================
# DATASET
# ============================================================

class SkinCancerDataset(Dataset):

    def __init__(self, metadata, image_dir, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        image_id = self.metadata.iloc[idx]["image_id"]
        label = self.metadata.iloc[idx]["encoded_labels"]

        image_path = os.path.join(
            self.image_dir,
            image_id + ".jpg"
        )

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # OpenCV loads images as BGR.
        # Convert to RGB before passing to the model.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.ToTensor(),

    # Same normalization used during prediction
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(len(metadata))
labels = metadata["encoded_labels"].values

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_STATE
)

valid_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=labels[temp_indices],
    random_state=RANDOM_STATE
)

print("\nDataset split:")
print(f"Training:   {len(train_indices)}")
print(f"Validation: {len(valid_indices)}")
print(f"Testing:    {len(test_indices)}")


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = SkinCancerDataset(
    metadata,
    IMAGE_DIR,
    transform=train_transforms
)

eval_dataset = SkinCancerDataset(
    metadata,
    IMAGE_DIR,
    transform=test_transforms
)

train_subset = Subset(train_dataset, train_indices)
valid_subset = Subset(eval_dataset, valid_indices)
test_subset = Subset(eval_dataset, test_indices)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)


# ============================================================
# MODEL
# ============================================================

class ModifiedResNet50(nn.Module):

    def __init__(self, num_classes):

        super(ModifiedResNet50, self).__init__()

        self.model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = ModifiedResNet50(NUM_CLASSES)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

model = model.to(device)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None:

            self.best_loss = val_loss

        elif val_loss > self.best_loss + self.min_delta:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:

            self.best_loss = val_loss
            self.counter = 0


early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.005
)


# ============================================================
# TRAINING
# ============================================================

train_losses = []
valid_losses = []

train_accuracies = []
valid_accuracies = []

best_valid_accuracy = 0.0


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels_batch in valid_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            valid_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels_batch.size(0)

            correct += (
                predicted == labels_batch
            ).sum().item()

    valid_loss /= len(valid_loader)

    valid_accuracy = correct / total


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler.step(valid_loss)


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {valid_loss:.4f} | "
        f"Val Acc: {valid_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = valid_accuracy

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved.")


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping(valid_loss)

    if early_stopping.early_stop:

        print("Early stopping triggered.")
        break


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

os.makedirs(
    os.path.dirname(ENCODER_PATH),
    exist_ok=True
)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print("\nLabel encoder saved.")


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_labels = []
test_predictions = []

with torch.no_grad():

    for images, labels_batch in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        test_labels.extend(
            labels_batch.numpy()
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    valid_accuracies,
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()

plt.show()