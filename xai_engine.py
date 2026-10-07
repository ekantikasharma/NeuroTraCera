# xai_engine.py
# ==============================================================================
# Cerevix AI -> NeuroTraCera
# NeuroTraCera — Explainable AI (XAI) Engine
# Implements:
#   1. Grad-CAM
#   2. Grad-CAM++
#   3. LIME (Local Interpretable Model-agnostic Explanations)
#   4. SHAP (Kernel SHAP via Shapley Kernel)
#   5. Integrated Gradients
#   6. Cross-Method Agreement & Overlap Analysis
#
# Strictly operates on the trained ResNet50 model and uploaded MRI scan slices.
# ==============================================================================

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.special import binom
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

logger = logging.getLogger("cerevix.xai")


# ==============================================================================
# IMAGE UTILITIES & BASE64 CONVERSIONS
# ==============================================================================

def array_to_base64_png(arr: np.ndarray) -> str:
    """Converts a numpy RGB array (uint8) to a base64 Data URL."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_map(cam: np.ndarray) -> np.ndarray:
    """Safely normalizes a 2D attribution array to [0.0, 1.0]."""
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    c_min = float(cam.min())
    c_max = float(cam.max())
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)
    return np.clip(cam, 0.0, 1.0)


def create_colored_overlay(
    original_rgb: np.ndarray,
    attribution_map_224: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """
    Takes original image and 224x224 attribution map.
    Resizes map to original resolution, applies colormap, generates overlay,
    and extracts high-activation bounding box.
    """
    orig_h, orig_w = original_rgb.shape[:2]
    cam_norm = normalize_map(attribution_map_224)
    cam_full = cv2.resize(cam_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cam_full = np.clip(cam_full, 0.0, 1.0)

    # OpenCV Colormap mapping
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    cmap_id = cmap_map.get(str(colormap).lower(), cv2.COLORMAP_JET)

    heat = cv2.applyColorMap(np.uint8(cam_full * 255), cmap_id)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    overlay = (1.0 - alpha_clamped) * original_rgb.astype(np.float32) + alpha_clamped * heat.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Salient activation bounding box (85th percentile threshold)
    threshold = np.percentile(cam_full, 85)
    mask = (cam_full >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            bbox = {
                "x": int(bx),
                "y": int(by),
                "width": int(bw),
                "height": int(bh),
                "area": int(bw * bh),
            }
            # Draw boundary on overlay
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

    return heat, overlay, bbox


def get_target_layer(model: nn.Module, layer_name: str = "layer4") -> nn.Module:
    """Safely retrieves target ResNet50 layer."""
    mapping = {
        "layer1": getattr(model, "layer1", None),
        "layer2": getattr(model, "layer2", None),
        "layer3": getattr(model, "layer3", None),
        "layer4": getattr(model, "layer4", None),
    }
    layer = mapping.get(layer_name)
    if layer is None:
        layer = getattr(model, "layer4", None)
    return layer


# ==============================================================================
# 1. GRAD-CAM (CLASSIC)
# ==============================================================================

class GradCAM:
    """Standard Grad-CAM using first-order gradient averaging."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.f_hook = target_layer.register_forward_hook(self._f_hook)
        self.b_hook = target_layer.register_full_backward_hook(self._b_hook)

    def _f_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _b_hook(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            score = logits[:, class_index].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM activations/gradients could not be captured.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return normalize_map(cam)

    def remove_hooks(self):
        try:
            self.f_hook.remove()
        except Exception:
            pass
        try:
            self.b_hook.remove()
        except Exception:
            pass


def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_index: int,
    target_layer_name: str = "layer4",
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """Generates standard Grad-CAM visual attribution."""
    orig_rgb = np.array(image.convert("RGB"))
    x = transform(image.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
    layer = get_target_layer(model, target_layer_name)

    cam_obj = GradCAM(model, layer)
    try:
        raw_map = cam_obj.generate(x, class_index)
    finally:
        cam_obj.remove_hooks()

    heat_arr, overlay_arr, bbox = create_colored_overlay(orig_rgb, raw_map, alpha=alpha, colormap=colormap)

    mean_int = float(raw_map.mean())
    max_int = float(raw_map.max())
    cov_pct = float((raw_map >= 0.5).mean() * 100.0)

    return {
        "method": "Grad-CAM",
        "raw_map": raw_map,
        "heatmap_url": array_to_base64_png(heat_arr),
        "overlay_url": array_to_base64_png(overlay_arr),
        "bounding_box": bbox,
        "metrics": {
            "mean_intensity": mean_int,
            "max_intensity": max_int,
            "active_area_coverage_pct": cov_pct,
            "target_layer": target_layer_name,
        },
        "description": "Grad-CAM combines feature maps from the final convolutional stage with gradient-derived importance weights to visualize model attention.",
        "why_highlighted": "The highlighted regions represent areas of the convolutional feature representation that contributed strongly to the selected model output.",
    }


# ==============================================================================
# 2. GRAD-CAM++ (HIGHER-ORDER GRADIENT FORMULATION)
# ==============================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation (Chattopadhay et al., 2018).
    Uses 2nd and 3rd order gradient formulations for fine-grained localization
    and multi-instance object attribution.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.f_hook = target_layer.register_forward_hook(self._f_hook)
        self.b_hook = target_layer.register_full_backward_hook(self._b_hook)

    def _f_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _b_hook(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            score = logits[:, class_index].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM++ activations/gradients could not be captured.")

        grads = self.gradients
        acts = self.activations

        # Closed-form higher-order gradient weights
        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)
        spatial_sum = (acts * grad_3).sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grad_2 + spatial_sum + 1e-7
        alpha = grad_2 / denom

        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return normalize_map(cam)

    def remove_hooks(self):
        try:
            self.f_hook.remove()
        except Exception:
            pass
        try:
            self.b_hook.remove()
        except Exception:
            pass


def generate_gradcam_plus_plus(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_index: int,
    target_layer_name: str = "layer4",
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """Generates Grad-CAM++ attribution heatmap, overlay, and metrics."""
    orig_rgb = np.array(image.convert("RGB"))
    x = transform(image.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
    layer = get_target_layer(model, target_layer_name)

    cam_pp = GradCAMPlusPlus(model, layer)
    try:
        raw_map = cam_pp.generate(x, class_index)
    finally:
        cam_pp.remove_hooks()

    heat_arr, overlay_arr, bbox = create_colored_overlay(orig_rgb, raw_map, alpha=alpha, colormap=colormap)

    mean_int = float(raw_map.mean())
    max_int = float(raw_map.max())
    cov_pct = float((raw_map >= 0.5).mean() * 100.0)

    return {
        "method": "Grad-CAM++",
        "raw_map": raw_map,
        "heatmap_url": array_to_base64_png(heat_arr),
        "overlay_url": array_to_base64_png(overlay_arr),
        "bounding_box": bbox,
        "metrics": {
            "mean_intensity": mean_int,
            "max_intensity": max_int,
            "active_area_coverage_pct": cov_pct,
            "target_layer": target_layer_name,
        },
        "description": "Grad-CAM++ is an extension of Grad-CAM that uses higher-order gradient information to provide a more detailed estimate of which image regions contributed to a model prediction.",
        "why_highlighted": "The highlighted regions represent areas receiving stronger attribution under the Grad-CAM++ calculation for the selected prediction.",
    }


# ==============================================================================
# 3. LIME (LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS)
# ==============================================================================

def generate_lime(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_index: int,
    num_samples: int = 50,
    n_segments: int = 35,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """
    Generates authentic LIME superpixel perturbations and attribution map.
    Fits a weighted Ridge regression on superpixel presence vs target class probability.
    """
    orig_rgb = np.array(image.convert("RGB"))
    np_img = np.array(image.convert("RGB").resize((224, 224))) / 255.0

    # 1. Superpixel Segmentation
    segments = slic(np_img, n_segments=n_segments, compactness=10.0, sigma=1.0, start_label=0)
    num_segments = int(np.max(segments) + 1)

    # 2. Binary Perturbations
    rng = np.random.default_rng(42)
    perturbations = rng.integers(0, 2, size=(num_samples, num_segments))
    perturbations[0, :] = 1  # Ensure original image is evaluated

    # 3. Batch Image Construction
    batch_tensors = []
    for i in range(num_samples):
        p_mask = perturbations[i]
        sample_img = np_img.copy()
        inactive_segs = np.where(p_mask == 0)[0]
        if len(inactive_segs) > 0:
            mask = np.isin(segments, inactive_segs)
            sample_img[mask] = 0.0  # Black background baseline
        p_pil = Image.fromarray((sample_img * 255).astype(np.uint8))
        batch_tensors.append(transform(p_pil))

    batch_tensor = torch.stack(batch_tensors).to(device)

    # 4. Model Inference
    with torch.no_grad():
        batch_logits = model(batch_tensor)
        batch_probs = torch.softmax(batch_logits, dim=1)[:, class_index].cpu().numpy()

    # 5. Distance Kernel & Ridge Regression
    dists = np.linalg.norm(perturbations - 1.0, axis=1)
    kernel_width = 0.25 * np.sqrt(num_segments)
    weights = np.exp(-(dists ** 2) / (kernel_width ** 2 + 1e-8))

    ridge = Ridge(alpha=1.0)
    ridge.fit(perturbations, batch_probs, sample_weight=weights)
    sp_weights = ridge.coef_

    # 6. Reconstruct 2D Superpixel Attribution Map
    lime_map_raw = np.zeros((224, 224), dtype=np.float32)
    for seg_id in range(num_segments):
        lime_map_raw[segments == seg_id] = sp_weights[seg_id]

    pos_mask = (lime_map_raw > 0).astype(np.float32)
    neg_mask = (lime_map_raw < 0).astype(np.float32)

    # Normalized positive attribution for unified comparison
    raw_map = normalize_map(np.maximum(lime_map_raw, 0.0))

    # Visualizations:
    # A. Superpixel boundaries on original MRI
    sp_boundaries = orig_rgb.copy()
    seg_full = cv2.resize(segments.astype(np.int32), (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    edges = cv2.Canny(seg_full.astype(np.uint8), 0, 1)
    sp_boundaries[edges > 0] = [255, 255, 0]  # Yellow superpixel boundaries

    # B. Positive contributing regions overlay (Green)
    pos_full = cv2.resize(pos_mask, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    pos_overlay = orig_rgb.copy()
    green_tint = np.array([34, 197, 94], dtype=np.uint8)
    pos_indices = pos_full > 0.5
    pos_overlay[pos_indices] = (
        0.45 * orig_rgb[pos_indices].astype(np.float32) + 0.55 * green_tint.astype(np.float32)
    ).astype(np.uint8)

    # C. Negative contributing regions overlay (Red)
    neg_full = cv2.resize(neg_mask, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    neg_overlay = orig_rgb.copy()
    red_tint = np.array([239, 68, 68], dtype=np.uint8)
    neg_indices = neg_full > 0.5
    neg_overlay[neg_indices] = (
        0.45 * orig_rgb[neg_indices].astype(np.float32) + 0.55 * red_tint.astype(np.float32)
    ).astype(np.uint8)

    # D. Continuous Colormap Overlay
    heat_arr, overlay_arr, bbox = create_colored_overlay(orig_rgb, raw_map, alpha=alpha, colormap=colormap)

    num_pos_segs = int((sp_weights > 0).sum())
    num_neg_segs = int((sp_weights < 0).sum())

    return {
        "method": "LIME",
        "raw_map": raw_map,
        "heatmap_url": array_to_base64_png(heat_arr),
        "overlay_url": array_to_base64_png(overlay_arr),
        "superpixel_boundaries_url": array_to_base64_png(sp_boundaries),
        "positive_regions_url": array_to_base64_png(pos_overlay),
        "negative_regions_url": array_to_base64_png(neg_overlay),
        "bounding_box": bbox,
        "metrics": {
            "total_superpixels": num_segments,
            "positive_contributing_segments": num_pos_segs,
            "negative_contributing_segments": num_neg_segs,
            "max_positive_weight": float(sp_weights.max()),
            "min_negative_weight": float(sp_weights.min()),
            "perturbation_samples": num_samples,
        },
        "description": "LIME explains an individual prediction by creating small variations of the input image and observing how the model's prediction changes. It then identifies image regions that locally contribute to the prediction.",
        "why_highlighted": "The highlighted image segments represent regions whose perturbation had a measurable effect on the local prediction.",
    }


# ==============================================================================
# 4. SHAP (KERNEL SHAP VIA SHAPLEY KERNEL)
# ==============================================================================

def generate_shap(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_index: int,
    num_samples: int = 50,
    n_segments: int = 35,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """
    Generates authentic Shapley Additive Explanations using the Shapley Kernel
    (Lundberg & Lee, NeurIPS 2017) applied to MRI superpixel partitions.
    """
    orig_rgb = np.array(image.convert("RGB"))
    np_img = np.array(image.convert("RGB").resize((224, 224))) / 255.0

    # 1. Superpixel Segmentation
    segments = slic(np_img, n_segments=n_segments, compactness=10.0, sigma=1.0, start_label=0)
    num_segments = int(np.max(segments) + 1)

    # 2. Permutation Sampling
    rng = np.random.default_rng(123)
    perturbations = rng.integers(0, 2, size=(num_samples, num_segments))
    perturbations[0, :] = 1  # Full image
    perturbations[1, :] = 0  # Baseline image

    # 3. Batch Image Construction
    batch_tensors = []
    for i in range(num_samples):
        p_mask = perturbations[i]
        sample_img = np_img.copy()
        inactive_segs = np.where(p_mask == 0)[0]
        if len(inactive_segs) > 0:
            mask = np.isin(segments, inactive_segs)
            sample_img[mask] = 0.0
        p_pil = Image.fromarray((sample_img * 255).astype(np.uint8))
        batch_tensors.append(transform(p_pil))

    batch_tensor = torch.stack(batch_tensors).to(device)

    # 4. Model Inference
    with torch.no_grad():
        batch_logits = model(batch_tensor)
        batch_probs = torch.softmax(batch_logits, dim=1)[:, class_index].cpu().numpy()

    # 5. Shapley Kernel Weights:
    # pi(z) = (M - 1) / ( (M choose |z|) * |z| * (M - |z|) )
    shap_weights = np.zeros(num_samples, dtype=np.float64)
    for i in range(num_samples):
        z_sum = int(np.sum(perturbations[i]))
        if z_sum == 0 or z_sum == num_segments:
            shap_weights[i] = 10000.0  # Large boundary weight
        else:
            comb = binom(num_segments, z_sum)
            shap_weights[i] = (num_segments - 1) / (comb * z_sum * (num_segments - z_sum) + 1e-12)

    # 6. Fit Linear Model with Shapley Kernel
    shap_reg = Ridge(alpha=0.01)
    shap_reg.fit(perturbations, batch_probs, sample_weight=shap_weights)
    shap_values = shap_reg.coef_

    # 7. 2D SHAP Attribution Map
    shap_map_raw = np.zeros((224, 224), dtype=np.float32)
    for seg_id in range(num_segments):
        shap_map_raw[segments == seg_id] = shap_values[seg_id]

    raw_map = normalize_map(np.maximum(shap_map_raw, 0.0))

    # Visualizations
    heat_arr, overlay_arr, bbox = create_colored_overlay(orig_rgb, raw_map, alpha=alpha, colormap=colormap)

    # Diverging Bipolar Map (Blue for negative, Red for positive)
    orig_h, orig_w = orig_rgb.shape[:2]
    shap_full = cv2.resize(shap_map_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    max_abs = max(float(np.abs(shap_full).max()), 1e-6)
    bipolar_norm = np.clip(shap_full / max_abs, -1.0, 1.0)

    bipolar_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    # Positive -> Red channel
    pos_idx = bipolar_norm > 0
    bipolar_img[pos_idx, 0] = np.uint8(bipolar_norm[pos_idx] * 255)
    # Negative -> Blue channel
    neg_idx = bipolar_norm < 0
    bipolar_img[neg_idx, 2] = np.uint8(-bipolar_norm[neg_idx] * 255)

    diverging_overlay = (
        0.5 * orig_rgb.astype(np.float32) + 0.5 * bipolar_img.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    pos_count = int((shap_values > 0).sum())
    neg_count = int((shap_values < 0).sum())

    return {
        "method": "SHAP",
        "raw_map": raw_map,
        "heatmap_url": array_to_base64_png(heat_arr),
        "overlay_url": array_to_base64_png(overlay_arr),
        "diverging_overlay_url": array_to_base64_png(diverging_overlay),
        "bounding_box": bbox,
        "metrics": {
            "total_features": num_segments,
            "positive_features": pos_count,
            "negative_features": neg_count,
            "max_positive_shapley": float(shap_values.max()),
            "min_negative_shapley": float(shap_values.min()),
            "mean_shapley_magnitude": float(np.abs(shap_values).mean()),
        },
        "description": "SHAP estimates how different parts of the input contribute to the model's prediction by comparing the model output with and without information from different regions of the image.",
        "why_highlighted": "The highlighted regions represent features with stronger contribution toward or away from the selected model output.",
    }


# ==============================================================================
# 5. INTEGRATED GRADIENTS (PATH ACCUMULATION)
# ==============================================================================

def generate_integrated_gradients(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    class_index: int,
    n_steps: int = 15,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Dict[str, Any]:
    """
    Generates authentic Integrated Gradients (Sundararajan et al., 2017)
    accumulating gradients along the straight-line path from black baseline to MRI.
    """
    orig_rgb = np.array(image.convert("RGB"))
    x_in = transform(image.convert("RGB")).unsqueeze(0).to(device)
    baseline = torch.zeros_like(x_in)

    # Accumulate gradients along path alpha in [0.1, 1.0]
    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps)
    accumulated_grads = torch.zeros_like(x_in)

    for a in alphas:
        interpolated = baseline + float(a) * (x_in - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(interpolated)
            score = logits[:, class_index].sum()
            score.backward()
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach()

    avg_grads = accumulated_grads / float(n_steps)
    attributions = (x_in - baseline) * avg_grads

    # Spatial magnitude across the 3 color channels
    attr_np = attributions.squeeze(0).cpu().numpy()  # [3, 224, 224]
    attr_spatial = np.linalg.norm(attr_np, axis=0)  # [224, 224]

    raw_map = normalize_map(attr_spatial)

    heat_arr, overlay_arr, bbox = create_colored_overlay(orig_rgb, raw_map, alpha=alpha, colormap=colormap)

    mean_int = float(raw_map.mean())
    max_int = float(raw_map.max())
    cov_pct = float((raw_map >= 0.5).mean() * 100.0)

    return {
        "method": "Integrated Gradients",
        "raw_map": raw_map,
        "heatmap_url": array_to_base64_png(heat_arr),
        "overlay_url": array_to_base64_png(overlay_arr),
        "bounding_box": bbox,
        "metrics": {
            "steps": n_steps,
            "mean_intensity": mean_int,
            "max_intensity": max_int,
            "active_area_coverage_pct": cov_pct,
            "baseline": "Uniform Zero / Black Baseline",
        },
        "description": "Integrated Gradients estimates how individual input features contributed to the prediction by accumulating gradients along a path from a baseline image to the actual MRI.",
        "why_highlighted": "The highlighted regions represent input areas receiving stronger attribution along the path from the baseline input to the MRI.",
    }


# ==============================================================================
# 6. CROSS-METHOD AGREEMENT & OVERLAP ANALYSIS
# ==============================================================================

def compute_cross_method_metrics(
    raw_maps_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Computes rigorous, objective comparisons between 2D normalized attribution maps:
      - Spatial Pearson correlation (r)
      - Cosine similarity
      - Binarized Salient Mask IoU (top 20% activation threshold)
      - Shared active area percentage
    """
    methods = list(raw_maps_dict.keys())
    if len(methods) < 2:
        return {
            "available": False,
            "message": "At least two XAI methods must be computed to perform cross-method agreement analysis.",
            "pairwise_comparisons": [],
            "summary": "Single method evaluated.",
        }

    pairwise = []
    correlations = []
    ious = []
    cosine_sims = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1_name = methods[i]
            m2_name = methods[j]
            map1 = normalize_map(raw_maps_dict[m1_name])
            map2 = normalize_map(raw_maps_dict[m2_name])

            # Flatten
            v1 = map1.flatten()
            v2 = map2.flatten()

            # 1. Pearson Correlation
            std1 = np.std(v1)
            std2 = np.std(v2)
            if std1 > 1e-8 and std2 > 1e-8:
                corr = float(np.corrcoef(v1, v2)[0, 1])
            else:
                corr = 0.0
            corr = float(np.clip(corr, -1.0, 1.0))
            correlations.append(corr)

            # 2. Cosine Similarity
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
            else:
                cos_sim = 0.0
            cosine_sims.append(cos_sim)

            # 3. Binarized Salient Mask IoU (top 20% activated pixels)
            th1 = np.percentile(map1, 80)
            th2 = np.percentile(map2, 80)
            mask1 = map1 >= th1
            mask2 = map2 >= th2

            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou = float(intersection / (union + 1e-8))
            ious.append(iou)

            # 4. Shared Active Area Overlap %
            min_area = min(mask1.sum(), mask2.sum())
            if min_area > 0:
                shared_pct = float((intersection / min_area) * 100.0)
            else:
                shared_pct = 0.0

            pairwise.append({
                "method_a": m1_name,
                "method_b": m2_name,
                "pearson_correlation": round(corr, 3),
                "cosine_similarity": round(cos_sim, 3),
                "salient_iou": round(iou, 3),
                "shared_active_overlap_pct": round(shared_pct, 1),
            })

    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_cos = float(np.mean(cosine_sims)) if cosine_sims else 0.0

    if mean_corr > 0.4:
        summary_desc = "Substantial spatial agreement across evaluated explanation maps."
    elif mean_corr > 0.15:
        summary_desc = "Moderate spatial overlap, reflecting complementary regional perspectives."
    else:
        summary_desc = "Distinct attribution patterns highlighting different aspects of model behavior."

    return {
        "available": True,
        "evaluated_methods": methods,
        "mean_correlation": round(mean_corr, 3),
        "mean_salient_iou": round(mean_iou, 3),
        "mean_cosine_similarity": round(mean_cos, 3),
        "summary": summary_desc,
        "pairwise_comparisons": pairwise,
    }
