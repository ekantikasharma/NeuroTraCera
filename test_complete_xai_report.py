# test_complete_xai_report.py
import io
import base64
import requests
from PIL import Image
from report_generator import build_mri_report, generate_report_pdf
from server import ensure_all_xai_for_analysis

def run_test():
    print("=== STEP 1: Create dummy MRI scan ===")
    img = Image.new("RGB", (224, 224), color=(60, 80, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Simulating the exact state of 'res' right after upload in Streamlit (only Grad-CAM and Grad-CAM++)
    initial_analysis = {
        "id": "test-res-upload-001",
        "predicted_class": "Glioma",
        "class_index": 0,
        "probability": 0.942,
        "uncertainty": 0.012,
        "confidence_interval": [0.92, 0.96],
        "original_image_url": data_url,
        "image_name": "glioma_axial_slice.png",
        "sequence": "T1-CE Axial",
        "anatomy": "Left Frontal Lobe",
        "patient_name": "Test Patient",
        "patient_id": "PT-12345",
        "patient_case_id": "CX-TEST-001",
        "gradcam": {
            "overlay_url": data_url,
            "heatmap_url": data_url,
            "metrics": {"target_layer": "layer4"}
        },
        "gradcam_plus_plus": {
            "overlay_url": data_url,
            "heatmap_url": data_url,
            "metrics": {"target_layer": "layer4"}
        },
        # LIME, SHAP, and Integrated Gradients are intentionally NONE (as happens on initial upload)
        "lime": None,
        "shap": None,
        "integrated_gradients": None
    }

    print("\n=== STEP 2: Test ensure_all_xai_for_analysis ===")
    enriched = ensure_all_xai_for_analysis(dict(initial_analysis))
    assert enriched.get("lime") is not None, "LIME was not generated!"
    assert enriched.get("shap") is not None, "SHAP was not generated!"
    assert enriched.get("integrated_gradients") is not None, "Integrated Gradients was not generated!"
    assert enriched["lime"].get("overlay_url"), "LIME has no overlay_url!"
    assert enriched["shap"].get("overlay_url"), "SHAP has no overlay_url!"
    assert enriched["integrated_gradients"].get("overlay_url"), "Integrated Gradients has no overlay_url!"
    print("  [PASS] All 5 XAI methods are now populated in analysis!")

    print("\n=== STEP 3: Test build_mri_report with all 5 XAI methods ===")
    rep = build_mri_report(enriched)
    xai_viz = rep.get("xai_visualizations", [])
    print(f"  Visualizations in report: {len(xai_viz)}")
    assert len(xai_viz) == 5, f"Expected 5 visualizations, got {len(xai_viz)}"

    expected = [
        (2, "Grad-CAM"),
        (3, "Grad-CAM++"),
        (4, "LIME"),
        (5, "SHAP"),
        (6, "Integrated Gradients"),
    ]
    for fig_num, method_name in expected:
        match = [x for x in xai_viz if x["method"] == method_name]
        assert len(match) == 1, f"Missing visualization for {method_name}"
        assert match[0]["figure_number"] == fig_num, f"Expected Fig {fig_num} for {method_name}, got {match[0]['figure_number']}"
        print(f"  [PASS] {method_name}: Figure {fig_num} -> {match[0]['caption']}")

    print("\n=== STEP 4: Test generate_report_pdf with all 5 XAI methods ===")
    pdf_bytes = generate_report_pdf(rep)
    assert pdf_bytes is not None and len(pdf_bytes) > 10000, "PDF generation failed or file too small"
    assert pdf_bytes[:4] == b"%PDF", "Header is not a valid PDF"
    print(f"  [PASS] PDF generated successfully with all 5 XAI figures embedded! ({len(pdf_bytes):,} bytes)")

    print("\n=== STEP 5: Test POST /api/generate-mri-report with partial analysis ===")
    # Send initial_analysis (missing lime/shap/ig) to endpoint to test auto-generation on server
    res_api = requests.post(
        "http://127.0.0.1:8000/api/generate-mri-report",
        json={"analysis_data": initial_analysis},
        timeout=180
    )
    assert res_api.status_code == 200, f"API failed with {res_api.status_code}: {res_api.text}"
    api_rep = res_api.json().get("report")
    api_xai = api_rep.get("xai_visualizations", [])
    print(f"  API generated report with {len(api_xai)} visualizations")
    assert len(api_xai) == 5, f"Expected 5 visualizations from API, got {len(api_xai)}"
    for x in api_xai:
        print(f"    - Figure {x['figure_number']}: {x['method']}")

    print("\n=== STEP 6: Test POST /api/generate-mri-report/pdf ===")
    res_pdf = requests.post(
        "http://127.0.0.1:8000/api/generate-mri-report/pdf",
        json={"report": api_rep},
        timeout=60
    )
    assert res_pdf.status_code == 200, f"PDF API failed: {res_pdf.status_code}"
    assert res_pdf.content[:4] == b"%PDF", "Server response is not a valid PDF"
    print(f"  [PASS] Server returned valid PDF! ({len(res_pdf.content):,} bytes)")

    print("\n=======================================================")
    print("ALL TESTS PASSED: REPORT INCLUDES ALL 5 XAI METHODS!")
    print("=======================================================")

if __name__ == "__main__":
    run_test()
