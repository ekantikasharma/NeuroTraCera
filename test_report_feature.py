# test_report_feature.py
import io
import base64
import requests
from PIL import Image
from report_generator import build_mri_report, generate_report_pdf, format_report_text

def run_tests():
    # 1. Prepare test dummy PNG (100x100 RGB image)
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64_str = base64.b64encode(buf.getvalue()).decode()
    data_url = f'data:image/png;base64,{b64_str}'

    print('=== TEST 1: COMPLETE XAI SUITE ===')
    analysis_full = {
        'id': 'analysis-full-001',
        'predicted_class': 'Glioma',
        'probability': 0.965,
        'uncertainty_std': 0.0085,
        'uncertainty_level': 'Low',
        'confidence_interval': [0.948, 0.982],
        'class_probabilities': {'Glioma': 0.965, 'Meningioma': 0.015, 'No Tumor': 0.005, 'Pituitary': 0.015},
        'original_image_url': data_url,
        'image_name': 'patient_brain_axial.png',
        'sequence': 'T1-CE Axial',
        'anatomy': 'Left Temporal Lobe',
        'cam_layer': 'layer4',
        'mc_passes_count': 25,
        'patient_name': 'Eleanor Vance',
        'patient_id': 'PT-99421',
        'patient_case_id': 'CX-2026-001',
        '_patient_age': 54,
        '_patient_gender': 'Female',
        'referring_physician': 'Dr. Marcus Brody, MD',
        'gradcam': {'overlay_url': data_url, 'heatmap_url': data_url},
        'gradcam_plus_plus': {'overlay_url': data_url, 'heatmap_url': data_url},
        'lime': {'overlay_url': data_url},
        'shap': {'overlay_url': data_url},
        'integrated_gradients': {'overlay_url': data_url},
    }

    rep1 = build_mri_report(analysis_full)
    assert rep1['patient_information']['patient_name'] == 'Eleanor Vance'
    assert rep1['patient_information']['patient_id'] == 'PT-99421'
    assert rep1['patient_information']['patient_age'] == '54 yrs'
    assert rep1['patient_information']['patient_gender'] == 'Female'
    assert rep1['patient_information']['referring_physician'] == 'Dr. Marcus Brody, MD'
    assert len(rep1['xai_visualizations']) == 5
    expected_methods = ['Grad-CAM', 'Grad-CAM++', 'LIME', 'SHAP', 'Integrated Gradients']
    for i, m in enumerate(expected_methods, 2):
        assert rep1['xai_visualizations'][i-2]['method'] == m
        assert rep1['xai_visualizations'][i-2]['figure_number'] == i
        print(f"  [PASS] {m}: Figure {i}")

    pdf1 = generate_report_pdf(rep1)
    assert len(pdf1) > 5000
    assert pdf1[:4] == b'%PDF'
    print('  [PASS] PDF Generated successfully, size:', len(pdf1), 'bytes')


    print('\n=== TEST 2: PARTIAL XAI SUITE (Only Grad-CAM, SHAP, Integrated Gradients) ===')
    analysis_partial = {
        'id': 'analysis-partial-002',
        'predicted_class': 'Meningioma',
        'probability': 0.892,
        'original_image_url': data_url,
        'gradcam': {'overlay_url': data_url},
        'shap': {'overlay_url': data_url},
        'integrated_gradients': {'overlay_url': data_url},
    }
    rep2 = build_mri_report(analysis_partial)
    assert len(rep2['xai_visualizations']) == 3
    for v in rep2['xai_visualizations']:
        assert v['method'] in ['Grad-CAM', 'SHAP', 'Integrated Gradients']
        print(f"  [PASS] Found: {v['method']} -> {v['caption']}")
    assert not any(v['method'] in ['LIME', 'Grad-CAM++'] for v in rep2['xai_visualizations'])
    print('  [PASS] LIME and Grad-CAM++ correctly omitted without placeholders')


    print('\n=== TEST 3: MISSING PATIENT INFORMATION ===')
    analysis_no_patient = {
        'id': 'analysis-empty-patient-003',
        'predicted_class': 'No Tumor',
        'probability': 0.991,
        'original_image_url': data_url,
        'gradcam': {'overlay_url': data_url},
    }
    rep3 = build_mri_report(analysis_no_patient)
    assert rep3['patient_information']['patient_name'] == 'Not available'
    assert rep3['patient_information']['patient_id'] == 'Not available'
    assert rep3['patient_information']['patient_age'] == 'Not available'
    assert rep3['patient_information']['patient_gender'] == 'Not available'
    assert rep3['patient_information']['referring_physician'] == 'Not available'
    print('  [PASS] Missing patient fields cleanly reported as "Not available"')


    print('\n=== TEST 4: FASTAPI BACKEND API INTEGRATION ===')
    r = requests.post('http://127.0.0.1:8000/api/generate-mri-report', json={'analysis_data': analysis_full}, timeout=30)
    assert r.status_code == 200, f'API failed with {r.status_code}'
    api_rep = r.json().get('report')
    assert api_rep is not None
    print('  [PASS] POST /api/generate-mri-report returned 200, report ID:', api_rep['id'])

    r_pdf = requests.post('http://127.0.0.1:8000/api/generate-mri-report/pdf', json={'report': api_rep}, timeout=30)
    assert r_pdf.status_code == 200
    assert r_pdf.headers.get('content-type') == 'application/pdf'
    assert r_pdf.content[:4] == b'%PDF'
    print('  [PASS] POST /api/generate-mri-report/pdf returned 200 PDF bytes:', len(r_pdf.content))

    r_get = requests.get(f"http://127.0.0.1:8000/api/reports/{api_rep['id']}", timeout=30)
    assert r_get.status_code == 200
    print('  [PASS] GET /api/reports/{id} returned 200')


    print('\n=== TEST 5: REPORT EDITING & REGENERATION ===')
    r_update = requests.put(
        f"http://127.0.0.1:8000/api/reports/{api_rep['id']}",
        json={'findings': 'Custom edited finding by clinician.', 'notes': 'Reviewed with oncology.'},
        timeout=10
    )
    assert r_update.status_code == 200
    updated_rep = r_update.json().get('report')
    assert updated_rep['findings'] == 'Custom edited finding by clinician.'
    assert updated_rep['report_version'] == 2
    print('  [PASS] PUT /api/reports/{id} updated findings & incremented version to 2')

    # Regenerate
    regen_rep = build_mri_report(analysis_full)
    assert regen_rep['original_mri']['image'] == analysis_full['original_image_url']
    assert len(regen_rep['xai_visualizations']) == 5
    print('  [PASS] Regeneration uses same analysis and XAI without re-inference')

    print('\n>>> ALL 5 TESTS PASSED WITH 100% SUCCESS! <<<')

if __name__ == '__main__':
    run_tests()
