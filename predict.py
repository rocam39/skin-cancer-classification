import os
import cv2
import joblib
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_skin_cancer_resnet50_model.pth"
ENCODER_PATH = "models/label_encoder.pkl"

NUM_CLASSES = 7


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# MODEL ARCHITECTURE
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
            nn.Linear(
                num_features,
                num_classes
            )
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# LOAD MODEL
# ============================================================

model = ModifiedResNet50(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    ENCODER_PATH
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CLASS INFORMATION
# ============================================================

class_details = {

    "nv": (
        "Melanocytic nevi",
        "Benign"
    ),

    "mel": (
        "Melanoma",
        "Malignant"
    ),

    "df": (
        "Dermatofibroma",
        "Benign"
    ),

    "akiec": (
        "Actinic keratoses and "
        "intraepithelial carcinoma / "
        "Bowen's disease",
        "Malignant"
    ),

    "bcc": (
        "Basal cell carcinoma",
        "Malignant"
    ),

    "bkl": (
        "Benign keratosis-like lesions",
        "Benign"
    ),

    "vasc": (
        "Vascular lesions",
        "Benign"
    )
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    if not os.path.exists(image_path):

        print(
            f"Image not found: {image_path}"
        )

        return None


    # Read image

    img = cv2.imread(image_path)

    if img is None:

        print(
            f"Failed to read image: {image_path}"
        )

        return None


    # OpenCV BGR → RGB

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(img)


    # Apply preprocessing

    img_input = transform(img)


    # Add batch dimension

    img_input = img_input.unsqueeze(0)


    # Move to GPU/CPU

    img_input = img_input.to(device)


    # Prediction

    with torch.no_grad():

        outputs = model(img_input)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()


    # Convert encoded label back to original label

    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = probabilities[
        0,
        predicted_index
    ].item()


    return (
        predicted_index,
        predicted_class,
        confidence
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter the path of the skin lesion image: "
    )


    result = predict_image(
        image_path
    )


    if result is not None:

        predicted_index, predicted_class, confidence = result


        print(
            f"\nPredicted class index: "
            f"{predicted_index}"
        )

        print(
            f"Predicted class code: "
            f"{predicted_class}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )


        if predicted_class in class_details:

            class_name, cancer_type = (
                class_details[predicted_class]
            )

            print(
                f"Predicted class: "
                f"{class_name}"
            )

            print(
                f"Classification: "
                f"{cancer_type}"
            )

        else:

            print(
                "Class information unavailable."
            )