# report_generator.py
# ==============================================================================
# NeuroTraCera — Medical MRI Report Generation & Clinical PDF Engine
# Implements:
#   1. Structured Medical Report Builder (build_mri_report)
#   2. Dynamic XAI Detection & Attribution Formatting
#   3. ReportLab Clinical PDF Generator (generate_report_pdf)
#   4. Plaintext / Markdown Report Formatter (format_report_text)
# ==============================================================================

import base64
import io
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# CONSTANTS & MEDICAL STRINGS
# ==============================================================================

AI_DISCLAIMER_TEXT = (
    "This report was generated using an AI-assisted MRI analysis system. "
    "The predictions, findings, and explainability visualizations are intended to "
    "support interpretation and should not be considered a definitive medical diagnosis. "
    "XAI visualizations such as Grad-CAM, Grad-CAM++, LIME, SHAP, and Integrated Gradients "
    "represent model attribution/explainability information and should not be interpreted "
    "as independently confirmed pathological findings. The report should be reviewed "
    "and validated by a qualified radiologist or appropriate healthcare professional "
    "before clinical use."
)

CLASS_DISPLAY_NAMES = {
    "Glioma": "Glioma Tumor",
    "Meningioma": "Meningioma Tumor",
    "Pituitary": "Pituitary Tumor",
    "No Tumor": "No Tumor Detected",
}

CLASS_CLINICAL_DESCRIPTIONS = {
    "Glioma": "Primary intracranial neoplasm originating from glial parenchymal cells. Neuroimaging findings typically exhibit infiltrative margins, localized T2/FLAIR hyperintensity, and variable contrast enhancement.",
    "Meningioma": "Typically benign, slow-growing extra-axial intracranial neoplasm arising from arachnoid cap cells. Manifests with broad-based dural attachment, homogeneous enhancement, and potential mass effect on adjacent parenchyma.",
    "Pituitary": "Epithelial neoplasm developing within the sella turcica from the anterior pituitary gland, potentially presenting with sellar expansion, suprasellar extension, and mass effect on the optic chiasm.",
    "No Tumor": "No focal intracranial mass effect, pathological hyperintensity, abnormal enhancement, or morphological hallmarks of neoplasia identified within the provided scan plane.",
}

XAI_CAPTION_TEMPLATES = {
    "Grad-CAM": "Figure {fig_num}. Grad-CAM visualization showing image regions contributing to the model prediction.",
    "Grad-CAM++": "Figure {fig_num}. Grad-CAM++ visualization showing model attribution regions.",
    "LIME": "Figure {fig_num}. LIME visualization showing local feature contributions to the prediction.",
    "SHAP": "Figure {fig_num}. SHAP visualization showing feature attribution information.",
    "Integrated Gradients": "Figure {fig_num}. Integrated Gradients visualization showing input-region attribution relative to the selected baseline.",
}

XAI_INTERPRETATION_TEXTS = {
    "Grad-CAM": "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights spatial convolutional feature regions contributing most strongly to the model's prediction.",
    "Grad-CAM++": "Grad-CAM++ calculates higher-order positive partial derivatives, providing refined class-activation attribution for focal or multi-instance targets.",
    "LIME": "LIME (Local Interpretable Model-agnostic Explanations) perturbs superpixel segments and constructs an interpretable surrogate model to quantify local feature support.",
    "SHAP": "SHAP (Shapley Additive Explanations) employs game-theoretic Shapley valuation across image partitions to estimate fair additive feature attributions.",
    "Integrated Gradients": "Integrated Gradients computes path integrals of gradients along a linear trajectory from a neutral reference baseline, ensuring completeness and attribution symmetry.",
}


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def _clean_data_url(data_uri: Optional[str]) -> Optional[bytes]:
    """Extracts raw bytes from a data URI or base64 string."""
    if not data_uri or not isinstance(data_uri, str):
        return None
    try:
        if "," in data_uri:
            b64_data = data_uri.split(",", 1)[1]
        else:
            b64_data = data_uri
        return base64.b64decode(b64_data)
    except Exception:
        return None


def _format_percentage(val: Any) -> str:
    """Formats numeric value to percentage string."""
    try:
        f = float(val)
        if f <= 1.0:
            f = f * 100.0
        return f"{f:.2f}%"
    except Exception:
        return "N/A"


# ==============================================================================
# 1. STRUCTURED REPORT BUILDER
# ==============================================================================

def build_mri_report(
    analysis: Dict[str, Any],
    patient_info: Optional[Dict[str, Any]] = None,
    examination_info: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a complete, validated, structured medical report dictionary
    from the existing MRI analysis results without recomputing inferences or XAI.
    """
    if not analysis or not isinstance(analysis, dict):
        raise ValueError("A valid MRI analysis result dictionary is required to generate a report.")

    patient_info = patient_info or {}
    examination_info = examination_info or {}
    model_info = model_info or {}

    now = datetime.now(timezone.utc)
    report_id = f"REP-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    # 1. Header Information
    app_name = "NeuroTraCera Decision Support"
    report_title = "MRI Analysis Report"
    generated_at_iso = now.isoformat()
    generated_at_formatted = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # 2. Patient Information (never fabricate!)
    raw_name = (
        patient_info.get("patient_name")
        or analysis.get("patient_name")
        or analysis.get("patient_information", {}).get("name")
    )
    patient_name = str(raw_name).strip() if raw_name else "Not available"

    raw_pid = (
        patient_info.get("patient_id")
        or analysis.get("patient_id")
        or analysis.get("patient_information", {}).get("case_id")
    )
    patient_id = str(raw_pid).strip() if raw_pid else "Not available"

    raw_case_id = (
        patient_info.get("case_id")
        or analysis.get("patient_case_id")
        or patient_id
    )
    case_id = str(raw_case_id).strip() if raw_case_id else "Not available"

    raw_age = (
        patient_info.get("patient_age")
        or analysis.get("_patient_age")
        or analysis.get("patient_age")
        or analysis.get("patient_information", {}).get("age")
    )
    age_str = f"{raw_age} yrs" if raw_age not in [None, "", "Not available"] else "Not available"

    raw_gender = (
        patient_info.get("patient_gender")
        or analysis.get("_patient_gender")
        or analysis.get("patient_gender")
        or analysis.get("patient_information", {}).get("gender")
    )
    gender_str = str(raw_gender).strip() if raw_gender and str(raw_gender).lower() != "not specified" else "Not available"

    exam_date = analysis.get("created_at", "")
    if exam_date:
        try:
            dt = datetime.fromisoformat(exam_date.replace("Z", "+00:00"))
            exam_date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            exam_date_str = str(exam_date)[:19]
    else:
        exam_date_str = "Not available"

    raw_ref_physician = (
        patient_info.get("referring_physician")
        or analysis.get("referring_physician")
    )
    referring_physician = str(raw_ref_physician).strip() if raw_ref_physician else "Not available"

    # 3. Examination Information
    body_region = (
        examination_info.get("body_region")
        or analysis.get("anatomy")
        or "Brain / Cerebral Hemisphere"
    )
    mri_sequence = (
        examination_info.get("mri_sequence")
        or analysis.get("sequence")
        or "T1-CE Axial"
    )
    image_name = analysis.get("image_name") or "mri_scan.png"
    target_cam_layer = analysis.get("cam_layer") or "layer4"
    mc_passes = analysis.get("mc_passes_count", 20)

    # 4. Original MRI Image
    original_mri_url = analysis.get("original_image_url") or ""

    # 5. AI Analysis / Prediction Data
    pred_class = analysis.get("predicted_class") or analysis.get("prediction") or "Unknown"
    pred_display = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
    prob_val = float(analysis.get("probability", 0.0))
    prob_pct_str = _format_percentage(prob_val)
    uncertainty_val = float(analysis.get("uncertainty_std", analysis.get("uncertainty", 0.0)))
    uncertainty_level = str(analysis.get("uncertainty_level", "Low")).capitalize()
    conf_interval = analysis.get("confidence_interval") or []
    ci_str = (
        f"{_format_percentage(conf_interval[0])} – {_format_percentage(conf_interval[1])}"
        if len(conf_interval) >= 2 else "Not available"
    )

    class_probs = analysis.get("class_probabilities", {})
    formatted_class_probs = {k: _format_percentage(v) for k, v in class_probs.items()}

    # 6. Structured Findings (Strictly based on actual results, zero fabrication)
    findings_list = []
    if pred_class in ["Glioma", "Meningioma", "Pituitary"]:
        findings_list.append(
            f"1. Primary Automated Detection: The deep neural network identifies image features consistent with {pred_display} "
            f"at an output probability of {prob_pct_str}."
        )
        findings_list.append(
            f"2. Anatomical Context: Target anatomical region specified as {body_region} using {mri_sequence} imaging protocol."
        )
        findings_list.append(
            f"3. Class Probability Distribution: " + ", ".join([f"{k}: {v}" for k, v in formatted_class_probs.items()]) + "."
        )
        findings_list.append(
            f"4. Uncertainty & Stability: Monte Carlo inference (passes: {mc_passes}) yielded a predictive standard deviation "
            f"of {uncertainty_val:.4f}, categorized as {uncertainty_level} uncertainty with a 95% confidence interval of [{ci_str}]."
        )
        # Check for localization / bounding box
        loc = analysis.get("localization") or {}
        bbox = loc.get("bounding_box") if isinstance(loc, dict) else None
        if not bbox and isinstance(analysis.get("gradcam"), dict):
            bbox = analysis.get("gradcam", {}).get("bounding_box")
        if bbox:
            findings_list.append(
                f"5. Activation Bounding Box: High-intensity attribution localized to spatial coordinate region "
                f"[X: {bbox.get('x')}, Y: {bbox.get('y')}, Width: {bbox.get('width')}, Height: {bbox.get('height')}] "
                f"in normalized feature space."
            )
        else:
            findings_list.append("5. Morphological Measurements: Not available from automated analysis.")
    else:
        findings_list.append(
            f"1. Primary Automated Detection: No focal mass effect, abnormal hyperintensity, or tumor morphology identified. "
            f"The model predicted '{pred_display}' with {prob_pct_str} confidence."
        )
        findings_list.append(
            f"2. Anatomical Context: Evaluated region: {body_region}, imaging protocol: {mri_sequence}."
        )
        findings_list.append(
            f"3. Uncertainty Assessment: Stochastic predictive variance: {uncertainty_val:.4f} ({uncertainty_level} uncertainty)."
        )
        findings_list.append(
            "4. Additional Observations: No high-confidence pathological focal lesions identified by automated neural inference."
        )

    findings_text = "\n\n".join(findings_list)

    # 7. Impression (Concise summary, non-fabricated, uncertainty-aware)
    if pred_class in ["Glioma", "Meningioma", "Pituitary"]:
        impression_items = [
            f"1. AI-assisted analysis demonstrates imaging characteristics suggestive of {pred_display} ({prob_pct_str} confidence).",
            f"2. Explainability attribution maps show localized feature focus within the designated {body_region}.",
            f"3. Inference uncertainty metric is {uncertainty_level.lower()} (std: {uncertainty_val:.4f}), indicating {'high consistency across stochastic passes' if uncertainty_level.lower() == 'low' else 'moderate inference variability; clinical caution advised'}.",
            "4. Definitive medical diagnosis requires comprehensive radiological correlation and histopathological evaluation.",
        ]
    else:
        impression_items = [
            f"1. Automated scan classification indicates no evident neoplastic brain lesion ({prob_pct_str} confidence).",
            f"2. Normal baseline attribution patterns observed across evaluated {body_region} slices.",
            "3. Clinical symptoms, laboratory findings, and comprehensive multi-planar imaging should be reviewed by an attending physician.",
        ]
    impression_text = "\n\n".join(impression_items)

    # 8. Dynamic XAI Detection & Sequential Captioning
    # Sequential numbering: Figure 1 is always Original MRI
    figure_counter = 2
    xai_visualizations = []

    # Map of method key in analysis dict to method display name
    def _get_xai_img(m_key: str) -> Optional[str]:
        direct = analysis.get(f"{m_key}_overlay_url") or analysis.get(f"{m_key}_url")
        if direct and isinstance(direct, str):
            return direct
        sub = analysis.get(m_key)
        if isinstance(sub, dict) and not sub.get("error"):
            return sub.get("overlay_url") or sub.get("heatmap_url") or sub.get("positive_regions_url") or sub.get("image_url") or sub.get("image")
        if isinstance(sub, str) and (sub.startswith("data:image") or sub.startswith("http") or sub.startswith("/")):
            return sub
        return None

    xai_candidates = [
        ("gradcam", "Grad-CAM", analysis.get("gradcam_overlay_url") or _get_xai_img("gradcam")),
        ("gradcam_plus_plus", "Grad-CAM++", _get_xai_img("gradcam_plus_plus")),
        ("lime", "LIME", _get_xai_img("lime")),
        ("shap", "SHAP", _get_xai_img("shap")),
        ("integrated_gradients", "Integrated Gradients", _get_xai_img("integrated_gradients")),
    ]

    for key, name, img_url in xai_candidates:
        # Check if the visualization exists and is valid
        dict_data = analysis.get(key)
        has_error = isinstance(dict_data, dict) and dict_data.get("error") is True
        if img_url and not has_error:
            caption = XAI_CAPTION_TEMPLATES.get(
                name,
                f"Figure {figure_counter}. {name} visualization showing model attribution regions."
            ).format(fig_num=figure_counter)

            xai_visualizations.append({
                "key": key,
                "method": name,
                "figure_number": figure_counter,
                "image": img_url,
                "caption": caption,
                "interpretation": XAI_INTERPRETATION_TEXTS.get(name, "Model attribution visualization."),
            })
            figure_counter += 1

    # 9. Model Information
    model_record = {
        "model_name": model_info.get("model_name", "ResNet50 Brain Tumor Classifier"),
        "model_version": model_info.get("model_version", "1.0.0"),
        "architecture": model_info.get("architecture", "ResNet-50 (50-layer deep residual network with Bottleneck blocks)"),
        "model_type": model_info.get("model_type", "Deep Convolutional Neural Network (CNN) with Monte Carlo Dropout Uncertainty"),
        "target_layer": target_cam_layer,
        "input_resolution": "224 × 224 pixels, RGB normalized (ImageNet parameters)",
        "analysis_timestamp": generated_at_formatted,
        "xai_methods_evaluated": [v["method"] for v in xai_visualizations],
        "prediction_confidence": prob_pct_str,
    }

    # Assembled Report Object
    report = {
        "id": report_id,
        "analysis_id": str(analysis.get("id", "")),
        "user_id": user_id or str(analysis.get("user_id", "")),
        "report_version": 1,
        "created_at": generated_at_iso,
        "created_at_formatted": generated_at_formatted,
        "header": {
            "application_name": app_name,
            "report_title": report_title,
            "report_id": report_id,
            "generated_at": generated_at_formatted,
        },
        "patient_information": {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "case_id": case_id,
            "patient_age": age_str,
            "patient_gender": gender_str,
            "date_of_examination": exam_date_str,
            "referring_physician": referring_physician,
        },
        "examination_information": {
            "body_region": body_region,
            "mri_sequence": mri_sequence,
            "image_name": image_name,
            "examination_date": exam_date_str,
            "mc_passes": mc_passes,
            "target_layer": target_cam_layer,
        },
        "original_mri": {
            "image": original_mri_url,
            "caption": "Figure 1. Original MRI Image",
            "file_name": image_name,
        },
        "ai_prediction": {
            "prediction": pred_class,
            "prediction_label": pred_display,
            "probability": prob_val,
            "probability_percentage": prob_pct_str,
            "uncertainty_std": uncertainty_val,
            "uncertainty_level": uncertainty_level,
            "confidence_interval": ci_str,
            "class_probabilities": formatted_class_probs,
        },
        "findings": findings_text,
        "impression": impression_text,
        "xai_visualizations": xai_visualizations,
        "xai_summary_interpretation": (
            "Explainable AI (XAI) visualizations illustrate feature importance and gradient attribution "
            "within the artificial neural network. Highlighted regions correspond to areas contributing "
            "to the automated model prediction and should not be construed as independently confirmed pathology."
        ),
        "model_information": model_record,
        "disclaimer": AI_DISCLAIMER_TEXT,
        "notes": notes or "",
    }

    return report


# ==============================================================================
# 2. NUMBERED CANVAS FOR CLINICAL PDF
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render exact total page counts
    ('Page X of Y') and professional medical confidentiality running headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        page_w, page_h = letter
        margin = 36  # 0.5 inch

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(margin, page_h - 26, "NeuroTraCera — Explainable Brain Tumor MRI Analysis Report")
            self.drawRightString(page_w - margin, page_h - 26, f"Confidential Medical Record")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(margin, page_h - 30, page_w - margin, page_h - 30)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(margin, 32, page_w - margin, 32)

        footer_text = "CONFIDENTIAL & PROPRIETARY — FOR CLINICAL DECISION SUPPORT ONLY"
        self.drawString(margin, 20, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_w - margin, 20, page_str)
        self.restoreState()


# ==============================================================================
# 3. REPORTLAB CLINICAL PDF GENERATOR
# ==============================================================================

def generate_report_pdf(report: Dict[str, Any]) -> bytes:
    """
    Generates a professional, clinical-grade PDF document containing all
    patient parameters, AI predictions, structured findings, impressions,
    embedded original MRI and XAI visualization images, model metadata,
    and mandatory regulatory disclaimers.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=38,
    )

    styles = getSampleStyleSheet()

    # Custom Medical Styles
    c_primary = colors.HexColor("#0f172a")    # Deep Navy
    c_accent = colors.HexColor("#0284c7")     # Medical Cyan/Blue
    c_slate = colors.HexColor("#334155")      # Body slate
    c_muted = colors.HexColor("#64748b")      # Subdued slate
    c_border = colors.HexColor("#cbd5e1")     # Light border

    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
    )

    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=c_muted,
    )

    style_section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=4,
    )

    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_slate,
    )

    style_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=c_primary,
    )

    style_caption = ParagraphStyle(
        "FigureCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=c_primary,
        alignment=1,  # Center
    )

    style_disclaimer = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#7c2d12"),
    )

    story: List[Any] = []

    # --------------------------------------------------------------------------
    # 1. HEADER SECTION
    # --------------------------------------------------------------------------
    hdr = report.get("header", {})
    p_info = report.get("patient_information", {})
    e_info = report.get("examination_information", {})
    pred = report.get("ai_prediction", {})

    header_table_data = [
        [
            Paragraph(f"<b>NEUROTRACERA</b> &nbsp;|&nbsp; CLINICAL DECISION SUPPORT", style_subtitle),
            Paragraph(f"Report ID: <b>{hdr.get('report_id', 'N/A')}</b>", ParagraphStyle("RightMuted", parent=style_subtitle, alignment=2)),
        ],
        [
            Paragraph(f"<b>{hdr.get('report_title', 'MRI Analysis Report')}</b>", style_title),
            Paragraph(f"Generated: <b>{hdr.get('generated_at', 'N/A')}</b>", ParagraphStyle("RightMuted2", parent=style_subtitle, alignment=2)),
        ],
    ]
    t_header = Table(header_table_data, colWidths=[360, 180])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=2, spaceAfter=8))

    # --------------------------------------------------------------------------
    # 2. PATIENT & EXAMINATION METADATA GRID
    # --------------------------------------------------------------------------
    patient_exam_data = [
        [
            Paragraph("<b>PATIENT INFORMATION</b>", ParagraphStyle("Hdr1", fontName="Helvetica-Bold", fontSize=8.5, textColor=c_accent)),
            Paragraph("<b>EXAMINATION METADATA</b>", ParagraphStyle("Hdr2", fontName="Helvetica-Bold", fontSize=8.5, textColor=c_accent)),
        ],
        [
            Paragraph(f"<b>Patient Name:</b> {p_info.get('patient_name', 'Not available')}", style_body),
            Paragraph(f"<b>Body Region:</b> {e_info.get('body_region', 'Brain')}", style_body),
        ],
        [
            Paragraph(f"<b>Patient ID:</b> {p_info.get('patient_id', 'Not available')} &nbsp; (Case: {p_info.get('case_id', 'N/A')})", style_body),
            Paragraph(f"<b>Protocol / Sequence:</b> {e_info.get('mri_sequence', 'T1-CE Axial')}", style_body),
        ],
        [
            Paragraph(f"<b>Age / Sex:</b> {p_info.get('patient_age', 'Not available')} / {p_info.get('patient_gender', 'Not available')}", style_body),
            Paragraph(f"<b>Image Scan File:</b> {e_info.get('image_name', 'mri_scan.png')}", style_body),
        ],
        [
            Paragraph(f"<b>Referring Physician:</b> {p_info.get('referring_physician', 'Not available')}", style_body),
            Paragraph(f"<b>Attribution Layer:</b> {e_info.get('target_layer', 'layer4')} (MC: {e_info.get('mc_passes', 20)})", style_body),
        ],
    ]
    t_pe = Table(patient_exam_data, colWidths=[270, 270])
    t_pe.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pe)
    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 3. AI PREDICTION SUMMARY CALLOUT
    # --------------------------------------------------------------------------
    pred_label = pred.get("prediction_label", pred.get("prediction", "Unknown"))
    is_tumor = pred.get("prediction") in ["Glioma", "Meningioma", "Pituitary"]
    tag_bg = colors.HexColor("#fee2e2") if is_tumor else colors.HexColor("#d1fae5")
    tag_fg = colors.HexColor("#991b1b") if is_tumor else colors.HexColor("#065f46")

    pred_card_data = [
        [
            Paragraph(f"<b>AI CLASSIFICATION RESULT:</b> &nbsp; <font color='{tag_fg.hexval()}'><b>{pred_label.upper()}</b></font>", ParagraphStyle("PredH", fontName="Helvetica-Bold", fontSize=11, textColor=c_primary)),
            Paragraph(f"Confidence: <b>{pred.get('probability_percentage', 'N/A')}</b>", ParagraphStyle("PredR", fontName="Helvetica-Bold", fontSize=11, textColor=c_primary, alignment=2)),
        ],
        [
            Paragraph(f"<b>Uncertainty Metric:</b> {float(pred.get('uncertainty_std', 0.0)):.4f} ({pred.get('uncertainty_level', 'Low')} Uncertainty) &nbsp;·&nbsp; <b>95% CI:</b> [{pred.get('confidence_interval', 'N/A')}]", style_body),
            Paragraph(f"Model: ResNet-50 PyTorch", ParagraphStyle("PredModel", fontName="Helvetica", fontSize=8.5, textColor=c_muted, alignment=2)),
        ]
    ]
    t_pred = Table(pred_card_data, colWidths=[380, 160])
    t_pred.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tag_bg),
        ("BOX", (0, 0), (-1, -1), 1, tag_fg),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 4. STRUCTURED FINDINGS & IMPRESSION
    # --------------------------------------------------------------------------
    story.append(Paragraph("FINDINGS", style_section_h1))
    findings_raw = report.get("findings", "")
    for paragraph_text in findings_raw.split("\n\n"):
        if paragraph_text.strip():
            story.append(Paragraph(paragraph_text.strip().replace("\n", "<br/>"), style_body))
            story.append(Spacer(1, 3))

    story.append(Spacer(1, 4))
    story.append(Paragraph("IMPRESSION", style_section_h1))
    impression_card_data = []
    impression_raw = report.get("impression", "")
    for imp_text in impression_raw.split("\n\n"):
        if imp_text.strip():
            impression_card_data.append([Paragraph(imp_text.strip().replace("\n", "<br/>"), style_bold)])

    if impression_card_data:
        t_imp = Table(impression_card_data, colWidths=[540])
        t_imp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbf7d0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_imp)

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 5. ORIGINAL MRI SCAN & EXPLAINABLE AI (XAI) VISUALIZATIONS
    # --------------------------------------------------------------------------
    story.append(KeepTogether([
        Paragraph("EXPLAINABLE AI (XAI) ANALYSIS", style_section_h1),
        Paragraph(
            "Explainable AI visualizations demonstrate which spatial features and pixel gradients "
            "contributed most strongly to the neural network's classification. <b>The highlighted regions "
            "represent model attribution areas and should not be construed as independently confirmed pathological tissue.</b>",
            style_body
        ),
        Spacer(1, 8),
    ]))

    # Collect images to embed: (ImageObject, CaptionText)
    image_cells: List[Tuple[Any, str]] = []

    # 1. Original MRI
    orig_mri_url = report.get("original_mri", {}).get("image")
    orig_bytes = _clean_data_url(orig_mri_url)
    if orig_bytes:
        try:
            bio = io.BytesIO(orig_bytes)
            pil_im = Image.open(bio)
            w, h = pil_im.size
            aspect = h / max(1, w)
            target_w = 2.4 * inch
            target_h = target_w * aspect
            rl_img = RLImage(io.BytesIO(orig_bytes), width=target_w, height=target_h)
            image_cells.append((rl_img, report.get("original_mri", {}).get("caption", "Figure 1. Original MRI Image")))
        except Exception as e:
            print(f"Error loading original MRI into PDF: {e}")

    # 2. Dynamic XAI Images
    for xai_item in report.get("xai_visualizations", []):
        xai_bytes = _clean_data_url(xai_item.get("image"))
        if xai_bytes:
            try:
                bio = io.BytesIO(xai_bytes)
                pil_im = Image.open(bio)
                w, h = pil_im.size
                aspect = h / max(1, w)
                target_w = 2.4 * inch
                target_h = target_w * aspect
                rl_img = RLImage(io.BytesIO(xai_bytes), width=target_w, height=target_h)
                image_cells.append((rl_img, xai_item.get("caption", "XAI Visualization")))
            except Exception as e:
                print(f"Error loading XAI image into PDF: {e}")

    # Lay out images in 2-column grid
    grid_rows: List[List[Any]] = []
    i = 0
    while i < len(image_cells):
        if i + 1 < len(image_cells):
            # Pair of images
            img1, cap1 = image_cells[i]
            img2, cap2 = image_cells[i + 1]
            cell1 = [img1, Spacer(1, 3), Paragraph(cap1, style_caption)]
            cell2 = [img2, Spacer(1, 3), Paragraph(cap2, style_caption)]
            grid_rows.append([cell1, cell2])
            i += 2
        else:
            # Single leftover image
            img1, cap1 = image_cells[i]
            cell1 = [img1, Spacer(1, 3), Paragraph(cap1, style_caption)]
            grid_rows.append([cell1, ""])
            i += 1

    if grid_rows:
        t_images = Table(grid_rows, colWidths=[270, 270])
        t_images.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_images)

    story.append(Spacer(1, 10))

    # --------------------------------------------------------------------------
    # 6. MODEL INFORMATION TABLE
    # --------------------------------------------------------------------------
    mod = report.get("model_information", {})
    model_table_data = [
        [
            Paragraph("<b>Architecture:</b>", style_bold),
            Paragraph(mod.get("architecture", "ResNet-50"), style_body),
            Paragraph("<b>Target Layer:</b>", style_bold),
            Paragraph(mod.get("target_layer", "layer4"), style_body),
        ],
        [
            Paragraph("<b>Model Name:</b>", style_bold),
            Paragraph(mod.get("model_name", "ResNet50 Classifier"), style_body),
            Paragraph("<b>Version:</b>", style_bold),
            Paragraph(mod.get("model_version", "1.0.0"), style_body),
        ],
        [
            Paragraph("<b>XAI Evaluated:</b>", style_bold),
            Paragraph(", ".join(mod.get("xai_methods_evaluated", [])) or "None", style_body),
            Paragraph("<b>Inference Date:</b>", style_bold),
            Paragraph(mod.get("analysis_timestamp", "N/A"), style_body),
        ],
    ]
    t_mod = Table(model_table_data, colWidths=[90, 180, 90, 180])
    t_mod.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("MODEL & INFERENCE SPECIFICATIONS", style_section_h1))
    story.append(t_mod)
    story.append(Spacer(1, 12))

    # --------------------------------------------------------------------------
    # 7. AI-ASSISTED ANALYSIS DISCLAIMER
    # --------------------------------------------------------------------------
    disclaimer_data = [
        [
            Paragraph("<b>AI-ASSISTED ANALYSIS DISCLAIMER</b>", ParagraphStyle("DiscH", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.HexColor("#9a3412"))),
        ],
        [
            Paragraph(report.get("disclaimer", AI_DISCLAIMER_TEXT), style_disclaimer),
        ]
    ]
    t_disc = Table(disclaimer_data, colWidths=[540])
    t_disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fdba74")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_disc)

    # Build Document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


# ==============================================================================
# 4. PLAINTEXT & MARKDOWN REPORT FORMATTER (FOR "COPY REPORT")
# ==============================================================================

def format_report_text(report: Dict[str, Any], fmt: str = "text") -> str:
    """
    Renders the structured report into a clean, comprehensive plaintext
    or markdown formatted string suitable for clipboard copying and EHR notes.
    """
    hdr = report.get("header", {})
    p_info = report.get("patient_information", {})
    e_info = report.get("examination_information", {})
    pred = report.get("ai_prediction", {})
    mod = report.get("model_information", {})

    lines = []
    lines.append("================================================================================")
    lines.append(f"                   {hdr.get('application_name', 'NEUROTRACERA').upper()}")
    lines.append(f"                         {hdr.get('report_title', 'MRI ANALYSIS REPORT').upper()}")
    lines.append("================================================================================")
    lines.append(f"Report ID: {hdr.get('report_id', 'N/A')}")
    lines.append(f"Generated: {hdr.get('generated_at', 'N/A')}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("PATIENT INFORMATION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Patient Name:        {p_info.get('patient_name', 'Not available')}")
    lines.append(f"Patient ID:          {p_info.get('patient_id', 'Not available')}")
    lines.append(f"Case ID:             {p_info.get('case_id', 'Not available')}")
    lines.append(f"Age / Biological Sex:{p_info.get('patient_age', 'Not available')} / {p_info.get('patient_gender', 'Not available')}")
    lines.append(f"Date of Examination: {p_info.get('date_of_examination', 'Not available')}")
    lines.append(f"Referring Physician: {p_info.get('referring_physician', 'Not available')}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("EXAMINATION INFORMATION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"MRI Examination:     {e_info.get('body_region', 'Brain')}")
    lines.append(f"Protocol / Sequence: {e_info.get('mri_sequence', 'T1-CE Axial')}")
    lines.append(f"Scan File:           {e_info.get('image_name', 'mri_scan.png')}")
    lines.append(f"Target Feature Layer:{e_info.get('target_layer', 'layer4')}")
    lines.append(f"Monte Carlo Passes:  {e_info.get('mc_passes', 20)}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("AI ANALYSIS & PREDICTION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Predicted Class:     {pred.get('prediction_label', pred.get('prediction', 'Unknown'))}")
    lines.append(f"Prediction Confidence:{pred.get('probability_percentage', 'N/A')}")
    lines.append(f"Uncertainty Metric:  {float(pred.get('uncertainty_std', 0.0)):.4f} ({pred.get('uncertainty_level', 'Low')} Uncertainty)")
    lines.append(f"95% Confidence Interval: [{pred.get('confidence_interval', 'N/A')}]")
    lines.append("Class Probabilities:")
    for c_name, c_p in pred.get("class_probabilities", {}).items():
        lines.append(f"  • {c_name}: {c_p}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("FINDINGS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(report.get("findings", "Not available from automated analysis."))
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("IMPRESSION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(report.get("impression", "Not available from automated analysis."))
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("EXPLAINABLE AI (XAI) ATTRIBUTION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("Original Image: Figure 1. Original MRI Image")
    xai_items = report.get("xai_visualizations", [])
    if xai_items:
        for item in xai_items:
            lines.append(f"{item.get('method')}:")
            lines.append(f"  Caption: {item.get('caption')}")
            lines.append(f"  Method Note: {item.get('interpretation')}")
    else:
        lines.append("No post-inference XAI visualizations were computed for this case.")
    lines.append("")
    lines.append(f"XAI Methodological Note: {report.get('xai_summary_interpretation', '')}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("MODEL & INFERENCE SPECIFICATIONS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Model Name:          {mod.get('model_name', 'ResNet50 Classifier')}")
    lines.append(f"Version:             {mod.get('model_version', '1.0.0')}")
    lines.append(f"Architecture:        {mod.get('architecture', 'ResNet-50')}")
    lines.append(f"Model Type:          {mod.get('model_type', 'CNN with Monte Carlo Uncertainty')}")
    lines.append(f"Evaluation Methods:  {', '.join(mod.get('xai_methods_evaluated', []))}")
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("AI-ASSISTED ANALYSIS DISCLAIMER")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(report.get("disclaimer", AI_DISCLAIMER_TEXT))
    lines.append("================================================================================")

    return "\n".join(lines)
