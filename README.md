# NeuroTraCera — Explainable Brain Tumor Analytics Platform

A clinical-grade academic research and decision-support interface for Explainable AI (XAI) brain tumor detection.

## Pages

- Home
- Brain Tumour Detection
- Prediction Result
- Patient History
- About XAI

## Run in VS Code

1. Open this folder in VS Code.
2. Open the terminal.
3. (Optional) Create a virtual environment:

   Windows:
   `python -m venv venv`
   `venv\Scripts\activate`

   macOS/Linux:
   `python3 -m venv venv`
   `source venv/bin/activate`

4. Install dependencies:

   `pip install -r requirements.txt`

5. Run:

   `streamlit run app.py`

6. Streamlit will open the website in your browser.

## IMPORTANT — Connect Your Real ML Model

The current detection page deliberately uses a random/demo prediction so the frontend works immediately.

Open:

`pages/1_Brain_Tumour_Detection.py`

Find:

`# ---------------- DEMO MODEL ----------------`

Replace that section with your trained model inference.

Typical flow:

```python
model = load_your_model(...)
processed = preprocess(image)
probabilities = model.predict(processed)[0]
predicted_index = probabilities.argmax()
predicted = classes[predicted_index]
confidence = float(probabilities[predicted_index])
```

For a real XAI implementation, generate a Grad-CAM heatmap from the same MRI image and display it on the Prediction Result page.

## Suggested project structure

```text
xai_brain_tumor_streamlit/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── your_model_here
│
├── utils/
│   ├── preprocessing.py
│   └── xai.py
│
└── pages/
    ├── 1_Brain_Tumour_Detection.py
    ├── 2_Prediction_Result.py
    ├── 3_Patient_History.py
    └── 4_About_XAI.py
```

## Medical note

This is an academic/project interface, not a medical diagnostic system. Any real-world clinical use requires appropriate validation, safety review, privacy/security controls, and qualified medical oversight.
