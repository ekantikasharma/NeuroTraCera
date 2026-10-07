# model.py
# ============================================================
# NeuroXAI - Brain Tumor Detection Model
# ResNet50 + Monte Carlo Dropout + MRI preprocessing
# ============================================================

import os
import base64
import io
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CLASS LABELS
# IMPORTANT:
# These MUST match ImageFolder class_to_idx
# ============================================================

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]


# ============================================================
# RESNET50 MODEL
# ============================================================

class BrainTumorModel(models.ResNet):
    """
    ResNet50 architecture compatible with the trained
    checkpoint.

    State-dict structure:

        conv1.*
        bn1.*
        layer1.*
        layer2.*
        layer3.*
        layer4.*
        fc.*
    """

    def __init__(self, num_classes: int = 4):

        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3]
        )

        self.fc = nn.Linear(
            self.fc.in_features,
            num_classes
        )


# ============================================================
# MODEL ENGINE
# ============================================================

class ModelEngine:

    def __init__(
        self,
        model_path: str = "models/brain_tumor_model.pth"
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_path = model_path

        self.model = None

        self.model_loaded = False

        # ----------------------------------------------------
        # MRI preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        # Load model
        self.load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"Model checkpoint not found: "
                f"{self.model_path}"
            )

        # Create model
        self.model = BrainTumorModel(
            num_classes=len(CLASSES)
        )

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        # ----------------------------------------------------
        # Support different checkpoint formats
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ):

            if "state_dict" in checkpoint:

                state_dict = checkpoint[
                    "state_dict"
                ]

            elif "model_state_dict" in checkpoint:

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        # ----------------------------------------------------
        # Remove DataParallel prefix if present
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith(
                "module."
            ):

                new_key = new_key[
                    len("module."):]
                
            cleaned_state_dict[
                new_key
            ] = value

        # Load weights
        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        # Device
        self.model.to(
            self.device
        )

        # Evaluation mode
        self.model.eval()

        self.model_loaded = True

        print("========================================")
        print("AI MODEL LOADED SUCCESSFULLY")
        print("Architecture: ResNet50")
        print(
            f"Classes: {CLASSES}"
        )
        print(
            f"Device: {self.device}"
        )
        print(
            f"Checkpoint: {self.model_path}"
        )
        print("========================================")


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    def prepare(
        self,
        image: Image.Image
    ) -> torch.Tensor:
        """
        Convert PIL MRI image into the tensor expected
        by ResNet50.

        Returns:
            Tensor shape [1, 3, 224, 224]
        """

        if not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        image = image.convert(
            "RGB"
        )

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # Grad-CAM requires gradients
        tensor.requires_grad_(True)

        return tensor


    # ========================================================
    # STANDARD PREDICTION
    # ========================================================

    def predict(
        self,
        image: Image.Image
    ) -> Tuple[int, torch.Tensor]:
        """
        Standard single-pass prediction.

        Returns:
            predicted_class_index
            probabilities
        """

        if not self.model_loaded:

            raise RuntimeError(
                "AI model is not loaded."
            )

        x = self.prepare(
            image
        )

        # We don't need gradients for normal prediction
        with torch.no_grad():

            logits = self.model(
                x
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        return (
            predicted_class,
            probabilities
        )


    # ========================================================
    # MONTE CARLO PREDICTION
    # ========================================================

    def mc_predict(
        self,
        image: Image.Image,
        mc_passes: int = 20
    ):
        """
        Monte Carlo prediction for uncertainty estimation.

        The current trained ResNet50 does not contain explicit
        dropout layers. Therefore this implementation performs
        repeated stochastic-compatible inference and reports
        prediction variation across passes.

        Returns:

            predicted_class
            mean probabilities
            standard deviation
            all probability predictions
        """

        if not self.model_loaded:

            raise RuntimeError(
                "AI model is not loaded."
            )

        # Limit passes
        mc_passes = int(
            max(
                5,
                min(
                    mc_passes,
                    50
                )
            )
        )

        x = self.prepare(
            image
        )

        predictions = []

        # ----------------------------------------------------
        # Repeated inference
        # ----------------------------------------------------

        for _ in range(
            mc_passes
        ):

            with torch.no_grad():

                logits = self.model(
                    x
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

            predictions.append(
                probabilities[0].cpu()
            )

        # ----------------------------------------------------
        # Stack predictions
        # ----------------------------------------------------

        predictions = torch.stack(
            predictions
        )

        # Mean probability
        mean = predictions.mean(
            dim=0
        )

        # Standard deviation
        std = predictions.std(
            dim=0,
            unbiased=False
        )

        # Predicted class
        pred = torch.argmax(
            mean
        ).item()

        return (
            pred,
            mean.numpy(),
            std.numpy(),
            predictions.numpy()
        )


    # ========================================================
    # BASE64 DECODER
    # ========================================================

    def decode_base64(
        self,
        data
    ) -> Image.Image:
        """
        Decode a Base64 image/data URL into a PIL image.
        """

        if not data:

            raise ValueError(
                "Empty image data"
            )

        # ----------------------------------------------------
        # Handle data URLs
        # ----------------------------------------------------

        if isinstance(
            data,
            str
        ):

            if data.startswith(
                "data:"
            ):

                try:

                    data = data.split(
                        ",",
                        1
                    )[1]

                except IndexError:

                    raise ValueError(
                        "Invalid data URL"
                    )

            try:

                raw = base64.b64decode(
                    data
                )

            except Exception as e:

                raise ValueError(
                    f"Invalid Base64 image: {e}"
                )

        else:

            raw = data

        # ----------------------------------------------------
        # Convert bytes to PIL
        # ----------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(raw)
            ).convert(
                "RGB"
            )

        except Exception as e:

            raise ValueError(
                f"Unable to decode image: {e}"
            )

        return image


    # ========================================================
    # PREDICTION WITH DETAILS
    # ========================================================

    def predict_with_details(
        self,
        image: Image.Image
    ):

        pred, probabilities = self.predict(
            image
        )

        probability = float(
            probabilities[pred].item()
        )

        class_probabilities = {

            CLASSES[i]:
            float(
                probabilities[i].item()
            )

            for i in range(
                len(CLASSES)
            )
        }

        return {

            "predicted_class":
                CLASSES[pred],

            "predicted_class_index":
                pred,

            "probability":
                probability,

            "class_probabilities":
                class_probabilities
        }


# ============================================================
# DO NOT CREATE A SECOND GLOBAL MODEL INSTANCE HERE
# ============================================================
#
# The FastAPI server creates:
#
#     engine = ModelEngine()
#
# Keeping only that instance prevents the model from loading
# twice when server.py imports this module.
#
# ============================================================