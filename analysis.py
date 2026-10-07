from typing import Optional, Dict, List, Tuple, Any
from pydantic import BaseModel, Field

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

class AnalysisRequest(BaseModel):
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_name: str = "mri_scan.png"
    mc_passes: int = Field(10, ge=1, le=50)
    cam_layer: str = "layer4"
    colormap: str = "jet"
    alpha: float = Field(0.55, ge=0.1, le=0.9)
    patient_case_id: Optional[str] = None
    sequence: str = "T1-CE Axial"
    anatomy: str = "Cerebral Hemisphere"
    run_all_xai: bool = False

class XAIExecutionRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    method: str = "all"  # "gradcam" | "gradcam_plus_plus" | "lime" | "shap" | "integrated_gradients" | "all"
    prediction_class_index: Optional[int] = None
    cam_layer: str = "layer4"
    colormap: str = "jet"
    alpha: float = 0.55
    analysis_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: str
    image_name: str
    prediction: str
    predicted_class: str
    class_index: int
    probability: float
    class_probabilities: Dict[str, float]
    uncertainty: float
    uncertainty_level: str
    confidence_interval: Tuple[float, float]
    mc_passes_count: int
    original_image_url: str
    gradcam_heatmap_url: str
    gradcam_overlay_url: str
    localization: Dict[str, object]
    xai_metrics: Dict[str, object]
    interpretation: str
    sequence: str
    anatomy: str
    gradcam_plus_plus: Optional[Dict[str, Any]] = None
    lime: Optional[Dict[str, Any]] = None
    shap: Optional[Dict[str, Any]] = None
    integrated_gradients: Optional[Dict[str, Any]] = None
    cross_method_analysis: Optional[Dict[str, Any]] = None
    disclaimer: str = (
        "Academic research prototype. Results describe model behavior and "
        "do not constitute a medical diagnosis or confirmed tumor boundary."
    )


class ReportGenerationRequest(BaseModel):
    analysis_id: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    patient_information: Optional[Dict[str, Any]] = None
    examination_information: Optional[Dict[str, Any]] = None
    model_information: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ReportUpdateRequest(BaseModel):
    findings: Optional[str] = None
    impression: Optional[str] = None
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    referring_physician: Optional[str] = None
    notes: Optional[str] = None


class ReportResponse(BaseModel):
    id: str
    analysis_id: str
    created_at: str
    report_content: Dict[str, Any]
    report_version: int = 1


class PDFGenerationRequest(BaseModel):
    report: Optional[Dict[str, Any]] = None
    analysis_id: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None


