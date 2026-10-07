# gradcam.py
# NeuroXAI - Grad-CAM Explainability and Localization

import base64
import io
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from model import CLASSES


class GradCAM:
    """
    Grad-CAM implementation for the NeuroXAI ResNet50 model.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Forward hook
        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        # Backward hook
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_index: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            x: Preprocessed image tensor [1, 3, 224, 224]
            class_index: Target class index

        Returns:
            Normalized CAM as numpy array [224, 224]
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B,C,H,W], got {x.shape}"
            )

        self.model.zero_grad(set_to_none=True)

        # Make sure gradients are enabled
        with torch.enable_grad():

            logits = self.model(x)

            if class_index < 0 or class_index >= logits.shape[1]:
                raise ValueError(
                    f"Invalid class index: {class_index}"
                )

            # Score for target class
            score = logits[:, class_index].sum()

            # Backpropagation
            score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # Global average pooling of gradients
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            weights * self.activations
        ).sum(dim=1)

        # ReLU
        cam = torch.relu(cam)

        # First image in batch
        cam = cam[0].detach().cpu().numpy()

        # Resize to model input resolution
        cam = cv2.resize(
            cam,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize between 0 and 1
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        """
        Remove PyTorch hooks when GradCAM is no longer needed.
        """

        try:
            self.forward_handle.remove()
        except Exception:
            pass

        try:
            self.backward_handle.remove()
        except Exception:
            pass


# ============================================================
# TARGET LAYER
# ============================================================

def get_target_layer(model, name: str = "layer4"):
    """
    Get the requested ResNet50 layer.

    IMPORTANT:
    BrainTumorModel directly inherits torchvision ResNet50,
    therefore layers are accessed as:

        model.layer1
        model.layer2
        model.layer3
        model.layer4

    NOT:

        model.backbone.layer4
        model.model.layer4
    """

    mapping = {
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }

    return mapping.get(
        name,
        model.layer4
    )


# ============================================================
# IMAGE ENCODING
# ============================================================

def image_to_base64(arr: np.ndarray) -> str:
    """
    Convert RGB numpy image to base64 PNG.
    """

    image = Image.fromarray(
        np.uint8(arr)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# HEATMAP + OVERLAY + LOCALIZATION
# ============================================================

def heatmap_and_overlay(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet"
):
    """
    Create:

    1. Grad-CAM heatmap
    2. MRI + Grad-CAM overlay
    3. Approximate high-activation bounding box
    4. XAI metrics

    NOTE:
    The bounding box is an approximate localization derived
    from Grad-CAM activation. It is NOT a medical segmentation
    mask.
    """

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    alpha = float(
        np.clip(alpha, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = np.array(
        image.convert("RGB")
    )

    height, width = original.shape[:2]

    # --------------------------------------------------------
    # Normalize CAM
    # --------------------------------------------------------

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min > 1e-8:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(cam)

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Resize CAM to original MRI size
    # --------------------------------------------------------

    cam_full = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    cam_full = np.clip(
        cam_full,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Select OpenCV colormap
    # --------------------------------------------------------

    colormap_name = str(
        colormap
    ).upper()

    if not colormap_name.startswith(
        "COLORMAP_"
    ):
        colormap_name = (
            "COLORMAP_"
            + colormap_name
        )

    cmap = getattr(
        cv2,
        colormap_name,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------------

    heat = cv2.applyColorMap(
        np.uint8(cam_full * 255),
        cmap
    )

    heat = cv2.cvtColor(
        heat,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        (1.0 - alpha) * original.astype(np.float32)
        + alpha * heat.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Approximate localization
    # --------------------------------------------------------

    # Top 15% activation threshold
    threshold = np.percentile(
        cam_full,
        85
    )

    mask = (
        cam_full >= threshold
    ).astype(np.uint8) * 255

    # Remove very small noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bbox: Optional[dict] = None

    if contours:

        # Ignore extremely small regions
        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > 20
        ]

        if valid_contours:

            largest = max(
                valid_contours,
                key=cv2.contourArea
            )

            x, y, bw, bh = cv2.boundingRect(
                largest
            )

            bbox = {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh),
                "area": int(bw * bh),
            }

            # Draw bounding box
            cv2.rectangle(
                overlay,
                (x, y),
                (x + bw, y + bh),
                (255, 255, 255),
                2
            )

    # --------------------------------------------------------
    # XAI metrics
    # --------------------------------------------------------

    active_area = float(
        (cam_full >= 0.5).mean()
    )

    mean_activation = float(
        cam_full.mean()
    )

    max_activation = float(
        cam_full.max()
    )

    std_activation = float(
        cam_full.std()
    )

    snr = float(
        mean_activation
        / (std_activation + 1e-8)
    )

    metrics = {
        "mean_heatmap_intensity": mean_activation,
        "max_heatmap_intensity": max_activation,
        "active_area_coverage": active_area,
        "snr": snr,

        # Ground-truth segmentation is not currently available
        "ground_truth_available": False,

        "iou": None,
        "dice": None,
    }

    # --------------------------------------------------------
    # Convert images to Base64
    # --------------------------------------------------------

    heatmap_base64 = image_to_base64(
        heat
    )

    overlay_base64 = image_to_base64(
        overlay
    )

    return (
        heatmap_base64,
        overlay_base64,
        bbox,
        metrics
    )