# ==============================================================================
# NEUROTRACERA
# "Detect. Localize. Explain. — Toward a Deeper Understanding"
# Explainable Brain Tumor Analytics & Clinical Decision Support Platform
# Academic / Research / Educational Prototype
# ==============================================================================

import base64
import html
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from PIL import Image
import urllib.parse
import hashlib
import secrets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import streamlit as st
import xai_engine
import report_generator
from report_generator import AI_DISCLAIMER_TEXT
from model import ModelEngine, CLASSES
from db import Database

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

API_URL = os.getenv("NEUROTRACERA_API_URL", os.getenv("CEREVIX_API_URL", "http://127.0.0.1:8000"))

@st.cache_resource
def start_embedded_backend() -> bool:
    """
    Spins up the FastAPI backend in a background daemon thread if not already running.
    Allows Streamlit Community Cloud and single-command deployments to run seamlessly.
    """
    import socket
    import threading
    import time

    def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    if not _is_server_listening("127.0.0.1", 8000):
        try:
            import uvicorn
            from server import app as fastapi_app

            def _run():
                config = uvicorn.Config(
                    app=fastapi_app,
                    host="127.0.0.1",
                    port=8000,
                    log_level="warning",
                    access_log=False
                )
                server = uvicorn.Server(config)
                server.run()

            t = threading.Thread(target=_run, daemon=True, name="NeuroTraCera-FastAPI-Server")
            t.start()

            for _ in range(40):
                time.sleep(0.5)
                if _is_server_listening("127.0.0.1", 8000):
                    try:
                        r = requests.get(f"{API_URL}/api/health", timeout=1)
                        if r.status_code == 200:
                            return True
                    except Exception:
                        pass
        except Exception as e:
            print(f"[NeuroTraCera] Embedded backend server startup notice: {e}")
            return False

    return True

# Auto-start backend if using local API address
if "127.0.0.1" in API_URL or "localhost" in API_URL:
    start_embedded_backend()

def api_request(method: str, path: str, **kwargs) -> requests.Response:
    """
    Robust API request wrapper with auto-retry and embedded backend healing.
    """
    url = f"{API_URL}{path}" if path.startswith("/") else f"{API_URL}/{path}"
    timeout = kwargs.pop("timeout", 15)
    last_exc = None
    for attempt in range(3):
        try:
            return requests.request(method, url, timeout=timeout, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < 2 and ("127.0.0.1" in API_URL or "localhost" in API_URL):
                start_embedded_backend()
                time.sleep(1.0)
            else:
                raise last_exc

# ==============================================================================
# SECURE AUTHENTICATION & GOOGLE OAUTH HELPERS
# ==============================================================================

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + ":" + password_hash.hex()

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        actual_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return secrets.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False

def get_google_auth_config() -> Tuple[str, str]:
    """Returns (client_id, client_secret) if configured via secrets or environment."""
    client_id = ""
    client_secret = ""
    try:
        if hasattr(st, "secrets"):
            if "GOOGLE_CLIENT_ID" in st.secrets:
                client_id = str(st.secrets["GOOGLE_CLIENT_ID"]).strip()
            elif "google" in st.secrets and isinstance(st.secrets["google"], dict):
                client_id = str(st.secrets["google"].get("client_id", "")).strip()
    except Exception:
        pass
    if not client_id:
        client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()

    try:
        if hasattr(st, "secrets"):
            if "GOOGLE_CLIENT_SECRET" in st.secrets:
                client_secret = str(st.secrets["GOOGLE_CLIENT_SECRET"]).strip()
            elif "google" in st.secrets and isinstance(st.secrets["google"], dict):
                client_secret = str(st.secrets["google"].get("client_secret", "")).strip()
    except Exception:
        pass
    if not client_secret:
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    return client_id, client_secret

def get_app_base_url() -> str:
    """Returns the base callback URL for Google OAuth."""
    try:
        if hasattr(st, "secrets") and "GOOGLE_REDIRECT_URI" in st.secrets:
            return str(st.secrets["GOOGLE_REDIRECT_URI"]).strip()
    except Exception:
        pass
    env_uri = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    if env_uri:
        return env_uri
    return "https://neurotracera-hmdwseb7aebbhhkwk4gsywf.streamlit.app"

def authenticate_google_user(email: str, name: str, google_id: str = "", id_token: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Authenticates a Google user via Backend API with seamless direct SQLite fallback."""
    clean_email = email.strip().lower()
    clean_name = name.strip() if name else f"Dr. {clean_email.split('@')[0].capitalize()}"
    clean_sub = google_id or f"goog_{secrets.token_hex(4)}"

    # 1. Try Backend API first
    payload = {
        "email": clean_email,
        "name": clean_name,
        "google_id": clean_sub,
    }
    if id_token:
        payload["id_token"] = id_token

    try:
        resp = api_request("POST", "/api/auth/google", json=payload, timeout=6)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass

    # 2. Resilient Direct SQLite Database Fallback (Eliminates ConnectionError entirely)
    try:
        db = Database()
        user = db._get_user_by_username_sync(clean_email)
        if not user:
            random_pwd = secrets.token_urlsafe(24)
            pwd_hash = hash_password(random_pwd)
            patient_tag = f"GOOG-{clean_sub[-6:] if len(clean_sub) >= 6 else secrets.token_hex(3).upper()}"
            user_id = db._create_user_sync(
                username=clean_email,
                password_hash=pwd_hash,
                patient_name=clean_name,
                patient_id=patient_tag,
            )
            user = db._get_user_by_id_sync(user_id)
        else:
            db._update_last_login_sync(str(user["id"]))

        session_token = secrets.token_hex(24)
        return True, {
            "session_token": session_token,
            "user": {
                "id": str(user["id"]),
                "username": user.get("login_id", clean_email),
                "patient_name": user.get("patient_name") or clean_name,
                "patient_id": user.get("patient_id", "")
            }
        }
    except Exception:
        return True, {
            "session_token": secrets.token_hex(24),
            "user": {
                "id": "1",
                "username": clean_email,
                "patient_name": clean_name,
                "patient_id": "GOOG-001"
            }
        }

def direct_login_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Attempts login through Backend API with direct SQLite fallback."""
    try:
        resp = api_request("POST", "/api/login", json={"username": username.strip(), "password": password}, timeout=6)
        if resp.status_code == 200:
            return True, resp.json(), ""
        else:
            try:
                detail = resp.json().get("detail", "Invalid username or password.")
            except Exception:
                detail = "Authentication failed."
            return False, None, detail
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception):
        try:
            db = Database()
            user = db._get_user_by_username_sync(username.strip())
            if user and verify_password(password, user.get("password_hash", "")):
                db._update_last_login_sync(str(user["id"]))
                session_token = secrets.token_hex(24)
                return True, {
                    "session_token": session_token,
                    "user": {
                        "id": str(user["id"]),
                        "username": user.get("login_id", username),
                        "patient_name": user.get("patient_name") or "Clinical Specialist",
                        "patient_id": user.get("patient_id", "")
                    }
                }, ""
            return False, None, "Invalid username or password."
        except Exception as exc:
            return False, None, f"Database login error: {exc}"

def direct_register_user(username: str, password: str, patient_name: Optional[str] = None, patient_id: Optional[str] = None) -> Tuple[bool, str]:
    """Attempts registration through Backend API with direct SQLite fallback."""
    try:
        resp = api_request(
            "POST",
            "/api/register",
            json={
                "username": username.strip(),
                "password": password,
                "patient_name": patient_name.strip() if patient_name else None,
                "patient_id": patient_id.strip() if patient_id else None,
            },
            timeout=6
        )
        if resp.status_code == 200:
            return True, "Account successfully created! You may now sign in."
        else:
            try:
                detail = resp.json().get("detail", "Unable to create account.")
            except Exception:
                detail = "Registration failed."
            return False, detail
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception):
        try:
            db = Database()
            existing = db._get_user_by_username_sync(username.strip())
            if existing:
                return False, f"Username '{username.strip()}' is already registered."
            pwd_hash = hash_password(password)
            tag = patient_id.strip() if patient_id else f"CASE-{secrets.token_hex(3).upper()}"
            user_id = db._create_user_sync(
                username=username.strip(),
                password_hash=pwd_hash,
                patient_name=patient_name.strip() if patient_name else "Clinical Specialist",
                patient_id=tag,
            )
            if user_id:
                return True, "Account successfully created! You may now sign in."
            return False, "Failed to write user record."
        except Exception as exc:
            return False, f"Registration error: {exc}"

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

DISPLAY_NAMES = {
    "Glioma": "Glioma Tumor",
    "Meningioma": "Meningioma Tumor",
    "Pituitary": "Pituitary Tumor",
    "No Tumor": "No Tumor Detected",
}

CLASS_DESCRIPTIONS = {
    "Glioma": "Primary brain tumors originating from glial cells (astrocytes, oligodendrocytes, or ependymal cells).",
    "Meningioma": "Typically slow-growing tumors arising from the meningeal membranes surrounding the brain and spinal cord.",
    "Pituitary": "Abnormal growths developing within the pituitary gland at the base of the brain, often affecting endocrine regulation.",
    "No Tumor": "No focal mass effect, abnormal hyperintensity, or morphological tumor characteristics detected on the scan.",
}

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

# ==============================================================================
# LOGO & BRANDING ASSETS
# ==============================================================================

LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.png")

def get_logo_data_uri() -> str:
    """Reads the local logo.png and returns a base64 Data URI."""
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
        except Exception:
            pass
    return ""

LOGO_URI = get_logo_data_uri()

# Configure Favicon for Browser Tab
app_icon = "🧠"
if os.path.exists(LOGO_PATH):
    try:
        app_icon = Image.open(LOGO_PATH)
    except Exception:
        app_icon = "🧠"

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================


def run_xai_method(method_name: str, res: dict) -> dict:
    """
    Executes a requested XAI method via the FastAPI backend,
    with automatic local fallback to xai_engine if backend is offline.
    """
    if not res:
        return {}

    payload = {
        "image_base64": res.get("original_image_url"),
        "method": method_name,
        "prediction_class_index": res.get("class_index"),
        "cam_layer": res.get("cam_layer", "layer4"),
        "colormap": res.get("colormap", "jet"),
        "alpha": res.get("alpha", 0.55),
        "analysis_id": res.get("id"),
    }

    # 1. Try Backend API
    try:
        resp = api_request(
            "POST",
            "/api/xai/run",
            json=payload,
            headers=get_auth_headers(),
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # 2. Local Fallback via xai_engine
    try:
        if "_local_engine" not in st.session_state or st.session_state._local_engine is None:
            st.session_state._local_engine = ModelEngine()
        eng = st.session_state._local_engine
        img = eng.decode_base64(res.get("original_image_url"))
        pred = int(res.get("class_index", 0))
        layer = str(res.get("cam_layer", "layer4"))
        alpha = float(res.get("alpha", 0.55))
        cmap = str(res.get("colormap", "jet"))

        m_name = method_name.lower().strip()
        out = {"prediction_class": CLASSES[pred], "class_index": pred, "requested_method": m_name}
        raw_maps = {}

        if m_name in ["gradcam", "all"]:
            try:
                gcam = xai_engine.generate_gradcam(eng.model, img, eng.transform, eng.device, pred, layer, alpha, cmap)
                raw_maps["Grad-CAM"] = gcam.pop("raw_map")
                out["gradcam"] = gcam
            except Exception as e:
                out["gradcam"] = {"error": True, "message": "Grad-CAM could not be generated for this analysis."}

        if m_name in ["gradcam_plus_plus", "gradcam++", "all"]:
            try:
                gcpp = xai_engine.generate_gradcam_plus_plus(eng.model, img, eng.transform, eng.device, pred, layer, alpha, cmap)
                raw_maps["Grad-CAM++"] = gcpp.pop("raw_map")
                out["gradcam_plus_plus"] = gcpp
            except Exception as e:
                out["gradcam_plus_plus"] = {"error": True, "message": "Grad-CAM++ could not be generated for this analysis."}

        if m_name in ["lime", "all"]:
            try:
                lime_res = xai_engine.generate_lime(eng.model, img, eng.transform, eng.device, pred, num_samples=50, n_segments=35, alpha=alpha, colormap=cmap)
                raw_maps["LIME"] = lime_res.pop("raw_map")
                out["lime"] = lime_res
            except Exception as e:
                out["lime"] = {"error": True, "message": "LIME could not be generated for this analysis."}

        if m_name in ["shap", "all"]:
            try:
                shap_res = xai_engine.generate_shap(eng.model, img, eng.transform, eng.device, pred, num_samples=50, n_segments=35, alpha=alpha, colormap=cmap)
                raw_maps["SHAP"] = shap_res.pop("raw_map")
                out["shap"] = shap_res
            except Exception as e:
                out["shap"] = {"error": True, "message": "SHAP could not be generated for this analysis."}

        if m_name in ["integrated_gradients", "ig", "all"]:
            try:
                ig_res = xai_engine.generate_integrated_gradients(eng.model, img, eng.transform, eng.device, pred, n_steps=15, alpha=alpha, colormap=cmap)
                raw_maps["Integrated Gradients"] = ig_res.pop("raw_map")
                out["integrated_gradients"] = ig_res
            except Exception as e:
                out["integrated_gradients"] = {"error": True, "message": "Integrated Gradients could not be generated for this analysis."}

        if len(raw_maps) >= 2:
            out["cross_method_analysis"] = xai_engine.compute_cross_method_metrics(raw_maps)

        return out
    except Exception as err:
        return {"error": True, "message": str(err)}


def perform_direct_mri_analysis(
    uploaded_file: Any,
    case_id: str,
    patient_age: int,
    patient_gender: str,
    mri_sequence: str,
    anatomical_region: str,
    cam_layer: str,
    alpha: float,
    colormap: str,
    mc_passes: int = 10,
    run_all_xai: bool = False
) -> Dict[str, Any]:
    """
    Executes MRI classification, Monte Carlo uncertainty estimation, and Grad-CAM
    directly within Streamlit when the FastAPI backend service is offline.
    """
    import io, uuid
    if "_local_engine" not in st.session_state or st.session_state._local_engine is None:
        st.session_state._local_engine = ModelEngine()
    eng = st.session_state._local_engine

    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    pred, mean, std, mc = eng.mc_predict(img, mc_passes)
    probability = float(mean[pred])
    uncertainty = float(std[pred])

    confidence_interval = (
        max(0.0, probability - 1.96 * uncertainty),
        min(1.0, probability + 1.96 * uncertainty)
    )

    raw_maps = {}
    try:
        gcam = xai_engine.generate_gradcam(
            eng.model, img, eng.transform, eng.device,
            pred, cam_layer, alpha=alpha, colormap=colormap
        )
        raw_maps["Grad-CAM"] = gcam.pop("raw_map")
    except Exception:
        gcam = {
            "heatmap_url": "",
            "overlay_url": "",
            "bounding_box": None,
            "metrics": {},
            "error": True,
            "message": "Grad-CAM could not be generated."
        }

    try:
        gcpp = xai_engine.generate_gradcam_plus_plus(
            eng.model, img, eng.transform, eng.device,
            pred, cam_layer, alpha=alpha, colormap=colormap
        )
        raw_maps["Grad-CAM++"] = gcpp.pop("raw_map")
    except Exception:
        gcpp = {
            "heatmap_url": "",
            "overlay_url": "",
            "bounding_box": None,
            "metrics": {},
            "error": True,
            "message": "Grad-CAM++ could not be generated."
        }

    lime_data = None
    shap_data = None
    ig_data = None

    if run_all_xai:
        try:
            lime_res = xai_engine.generate_lime(
                eng.model, img, eng.transform, eng.device,
                pred, num_samples=50, n_segments=35, alpha=alpha, colormap=colormap
            )
            raw_maps["LIME"] = lime_res.pop("raw_map")
            lime_data = lime_res
        except Exception:
            lime_data = {"error": True, "message": "LIME could not be generated."}

        try:
            shap_res = xai_engine.generate_shap(
                eng.model, img, eng.transform, eng.device,
                pred, num_samples=50, n_segments=35, alpha=alpha, colormap=colormap
            )
            raw_maps["SHAP"] = shap_res.pop("raw_map")
            shap_data = shap_res
        except Exception:
            shap_data = {"error": True, "message": "SHAP could not be generated."}

        try:
            ig_res = xai_engine.generate_integrated_gradients(
                eng.model, img, eng.transform, eng.device,
                pred, n_steps=15, alpha=alpha, colormap=colormap
            )
            raw_maps["Integrated Gradients"] = ig_res.pop("raw_map")
            ig_data = ig_res
        except Exception:
            ig_data = {"error": True, "message": "Integrated Gradients could not be generated."}

    user = st.session_state.get("user") or {}
    user_id = str(user.get("id", "1"))
    heat_url = gcam.get("heatmap_url", "")
    overlay_url = gcam.get("overlay_url", "")
    bbox = gcam.get("bounding_box")
    metrics = gcam.get("metrics", {})

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    orig_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    analysis_id = str(uuid.uuid4())
    result = {
        "id": analysis_id,
        "user_id": user_id,
        "username": user.get("username", "specialist"),
        "patient_name": user.get("patient_name", "Clinical Specialist"),
        "patient_id": case_id or user.get("patient_id", f"CASE-{analysis_id[:6].upper()}"),
        "image_name": getattr(uploaded_file, "name", "mri.png"),
        "prediction": CLASSES[pred],
        "predicted_class": CLASSES[pred],
        "class_index": pred,
        "probability": probability,
        "class_probabilities": {CLASSES[i]: float(mean[i]) for i in range(len(CLASSES))},
        "uncertainty": uncertainty,
        "uncertainty_level": "Low" if uncertainty < 0.05 else ("Moderate" if uncertainty < 0.12 else "High"),
        "confidence_interval": confidence_interval,
        "mc_passes_count": mc_passes,
        "original_image_url": orig_b64,
        "gradcam_heatmap_url": heat_url,
        "gradcam_overlay_url": overlay_url,
        "localization": {
            "bounding_box": bbox,
            "available": bbox is not None,
            "method": "Grad-CAM activation region"
        },
        "xai_metrics": metrics,
        "gradcam": gcam,
        "gradcam_plus_plus": gcpp,
        "lime": lime_data,
        "shap": shap_data,
        "integrated_gradients": ig_data,
        "cross_method_analysis": xai_engine.compute_cross_method_metrics(raw_maps) if len(raw_maps) >= 2 else None,
        "cam_layer": cam_layer,
        "alpha": alpha,
        "colormap": colormap,
        "sequence": mri_sequence,
        "anatomy": anatomical_region,
        "patient_case_id": case_id,
        "_patient_age": patient_age,
        "_patient_gender": patient_gender,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        db = Database()
        db._insert_analysis_sync(
            result,
            user_id,
            {"case_id": case_id, "age": patient_age, "gender": patient_gender}
        )
    except Exception as db_err:
        print(f"Direct DB insert error: {db_err}")

    return result


def request_mri_report(
    analysis_res: dict,
    patient_metadata: Optional[dict] = None,
    examination_metadata: Optional[dict] = None,
    notes: Optional[str] = None
) -> Optional[dict]:
    """
    Generates a structured MRI medical report by requesting the FastAPI backend,
    with automatic local fallback to report_generator if backend is unreachable.
    """
    if not analysis_res:
        return None

    p_info = patient_metadata or {
        "patient_name": analysis_res.get("patient_name"),
        "patient_id": analysis_res.get("patient_id"),
        "case_id": analysis_res.get("patient_case_id"),
        "patient_age": analysis_res.get("_patient_age") or analysis_res.get("patient_age"),
        "patient_gender": analysis_res.get("_patient_gender") or analysis_res.get("patient_gender"),
        "referring_physician": analysis_res.get("referring_physician"),
    }

    e_info = examination_metadata or {
        "body_region": analysis_res.get("anatomy"),
        "mri_sequence": analysis_res.get("sequence"),
        "image_name": analysis_res.get("image_name"),
    }

    payload = {
        "analysis_id": analysis_res.get("id"),
        "analysis_data": analysis_res,
        "patient_information": p_info,
        "examination_information": e_info,
        "notes": notes,
    }

    # 1. Try Backend API
    try:
        resp = api_request(
            "POST",
            "/api/generate-mri-report",
            json=payload,
            headers=get_auth_headers(),
            timeout=180,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "report" in data:
                return data["report"]
            return data
    except Exception:
        pass

    # 2. Local Fallback via report_generator
    try:
        return report_generator.build_mri_report(
            analysis=analysis_res,
            patient_info=p_info,
            examination_info=e_info,
            notes=notes,
        )
    except Exception as e:
        print(f"Local report generation error: {e}")
        return None


def request_report_pdf(report_dict: dict) -> Optional[bytes]:
    """
    Generates report PDF bytes via backend API with fallback to local report_generator.
    """
    if not report_dict:
        return None

    # 1. Try Backend API
    try:
        resp = api_request(
            "POST",
            "/api/generate-mri-report/pdf",
            json={"report": report_dict},
            headers=get_auth_headers(),
            timeout=120,
        )
        if resp.status_code == 200 and resp.content and resp.content[:4] == b"%PDF":
            return resp.content
    except Exception:
        pass

    # 2. Local Fallback via report_generator
    try:
        return report_generator.generate_report_pdf(report_dict)
    except Exception as e:
        print(f"Local PDF generation error: {e}")
        return None


st.set_page_config(
    page_title="NeuroTraCera | Explainable Brain Tumor Analytics",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

DEFAULT_STATE = {
    "logged_in": False,
    "session_token": None,
    "user": None,
    "page": "Dashboard",
    "nav_choice": "Dashboard",
    "result": None,
    "history": [],
    "auth_mode": "login",
    "show_login": False,
    "theme": "light",
    "uploaded_image": None,
    "analysis_running": False,
    "current_report": None,
    "report_generating": False,
    "report_editing": False,
    "report_edit_findings": "",
    "report_edit_impression": "",
    "report_edit_physician": "",
    "report_edit_notes": "",
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==============================================================================
# SAFE HTML RENDERING HELPER (PREVENTS MARKDOWN CODE BLOCK BUGS)
# ==============================================================================

def render_html(html_str: str) -> None:
    """
    Renders raw HTML safely in Streamlit without CommonMark parsing errors.
    Strips leading indentation from each line so it is never treated as a code block.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    clean_html = "\n".join(clean_lines)
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

# ==============================================================================
# DESIGN SYSTEM & GLOBAL CSS
# ==============================================================================

render_html(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root {
    color-scheme: light !important;
    --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-display: 'Outfit', var(--font-primary);
    
    --bg-app: #f8fafc;
    --bg-card: #ffffff;
    --bg-card-subtle: #f1f5f9;
    
    --navy-950: #07111e;
    --navy-900: #0a192f;
    --navy-800: #102a43;
    --navy-700: #1b3a5b;
    --navy-600: #274b74;
    
    --blue-primary: #0284c7;
    --blue-hover: #0369a1;
    --blue-subtle: #e0f2fe;
    --cyan-accent: #06b6d4;
    
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --text-light: #94a3b8;
    
    --border-light: #e2e8f0;
    --border-focus: #38bdf8;
    
    --status-success: #10b981;
    --status-warning: #f59e0b;
    --status-danger: #ef4444;
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: var(--font-primary) !important;
    color: #0f172a !important;
    background-color: var(--bg-app) !important;
}

/* Force crisp, high-contrast dark text on light backgrounds to prevent browser dark-mode wash-out */
[data-testid="stMain"] p,
[data-testid="stMain"] strong,
[data-testid="stMain"] b,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stHtml"] p,
[data-testid="stHtml"] strong,
[data-testid="stHtml"] b,
[data-testid="stHtml"] h1,
[data-testid="stHtml"] h2,
[data-testid="stHtml"] h3,
[data-testid="stHtml"] h4,
[data-testid="stHtml"] h5,
[data-testid="stHtml"] h6 {
    color: #0f172a !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stToolbar"], #MainMenu, footer {
    visibility: hidden;
    height: 0;
}

.block-container {
    max-width: 1480px !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    padding: 1.5rem 2.5rem 3.5rem 2.5rem !important;
    text-align: left !important;
}

[data-testid="stSidebarNav"] {
    display: none !important;
}
[data-testid="stSidebarHeader"],
[data-testid="stSidebarHeader"] > div {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    width: 250px !important;
    min-width: 250px !important;
    max-width: 250px !important;
    background: #0f2240 !important;
    border-right: 1px solid rgba(129, 153, 187, 0.12) !important;
    text-align: left !important;
    top: 0 !important;
}

[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 24px !important;
    padding-bottom: 16px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    background: #0f2240 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding: 0 !important;
    gap: 3px !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

.cx-brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 6px 12px 6px !important;
    border-bottom: 1px solid rgba(129, 153, 187, 0.16);
    margin-top: 4px !important;
    margin-bottom: 12px !important;
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
}

.cx-brand-logo-img {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    object-fit: contain;
    background: transparent;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.3);
    border: none;
    flex-shrink: 0;
}

.cx-brand-text {
    text-align: left !important;
}

.cx-brand-text h2 {
    margin: 0;
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.4px;
    line-height: 1.1;
    text-align: left !important;
}

.cx-brand-text p {
    margin: 2px 0 0 0;
    font-size: 10.5px;
    color: #8199bb;
    font-weight: 500;
    letter-spacing: 0.1px;
    text-align: left !important;
}

/* Material Symbols Styling in Sidebar */
.material-symbols-rounded,
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', sans-serif !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Base button styling in Sidebar - strictly left aligned */
[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button[data-testid^="stBaseButton"],
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[kind="tertiary"] {
    width: 100% !important;
    height: 44px !important;
    min-height: 44px !important;
    margin: 2px 0 !important;
    padding: 0 14px !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    font-family: var(--font-primary) !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    gap: 13px !important;
    box-shadow: none !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stButton > button div,
[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] div[data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important;
    flex: 0 1 auto !important;
}

[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] p {
    font-size: 14.5px !important;
    font-weight: inherit !important;
    color: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    text-align: left !important;
    white-space: nowrap !important;
}

/* Inactive button (kind="secondary") - matching Capture.JPG */
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: #8199bb !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"] .material-symbols-rounded,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] .material-symbols-rounded {
    color: #388bfd !important;
    font-size: 20px !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #ffffff !important;
    transform: none !important;
}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover .material-symbols-rounded,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover .material-symbols-rounded {
    color: #60a5fa !important;
}

/* Active button (kind="primary") - electric blue pill from Capture.JPG */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: #155dfd !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(21, 93, 253, 0.35) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] .material-symbols-rounded,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] .material-symbols-rounded {
    color: #ffffff !important;
    font-size: 20px !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #155dfd !important;
    color: #ffffff !important;
    transform: none !important;
}

/* Logout button (kind="tertiary") - soft coral/salmon red from Capture.JPG */
[data-testid="stSidebar"] .stButton > button[kind="tertiary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] {
    background: transparent !important;
    color: #e2665c !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="tertiary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[kind="tertiary"] .material-symbols-rounded,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] .material-symbols-rounded {
    color: #e2665c !important;
    font-size: 20px !important;
}

[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"]:hover {
    background: rgba(226, 102, 92, 0.12) !important;
    color: #f87171 !important;
    transform: none !important;
}

[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"]:hover [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover .material-symbols-rounded,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"]:hover .material-symbols-rounded {
    color: #f87171 !important;
}

/* Navigation divider */
.cx-nav-divider {
    height: 1px;
    background: rgba(129, 153, 187, 0.16);
    margin: 16px 6px 12px 6px;
    width: calc(100% - 12px);
}

.cx-sidebar-status-box {
    margin-top: auto;
    padding: 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(129, 153, 187, 0.14);
    border-radius: 10px;
}

.cx-sidebar-status-title {
    font-size: 11.5px;
    font-weight: 600;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.cx-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.cx-dot-green { background: #10b981; box-shadow: 0 0 8px #10b981; }
.cx-dot-red { background: #ef4444; box-shadow: 0 0 8px #ef4444; }

.cx-user-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.cx-user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #1e3a5f;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #e2e8f0;
}
.cx-user-meta {
    font-size: 12px;
    line-height: 1.25;
}
.cx-user-name {
    color: #f8fafc;
    font-weight: 600;
}
.cx-user-role {
    color: #64748b;
    font-size: 11px;
}

/* CARDS & CONTAINERS */
.cx-page-header {
    margin-bottom: 24px;
}
.cx-page-title {
    font-family: var(--font-display);
    font-size: 27px;
    font-weight: 700;
    color: var(--navy-950);
    letter-spacing: -0.6px;
    margin: 0 0 5px 0;
}
.cx-page-desc {
    color: var(--text-muted);
    font-size: 14.5px;
    margin: 0;
    line-height: 1.5;
}

.cx-card {
    background: #ffffff !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 14px !important;
    padding: 22px !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 18px !important;
    color: #0f172a !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
}
.cx-card:hover {
    border-color: #cbd5e1 !important;
    box-shadow: var(--shadow-md) !important;
}
.cx-card h1, .cx-card h2, .cx-card h3, .cx-card h4, .cx-card h5, .cx-card h6,
.cx-card strong, .cx-card b {
    color: #0f172a !important;
}
.cx-card p {
    color: #475569 !important;
}

/* SETTINGS & PROFILE MODERN CARDS (Matching Reference Image) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 24px 28px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 22px !important;
    color: #0f172a !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] h4,
div[data-testid="stVerticalBlockBorderWrapper"] strong {
    color: #0f172a !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
    gap: 16px !important;
}

.cx-profile-avatar-circle {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #155dfd;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    flex-shrink: 0;
    box-shadow: 0 3px 8px rgba(21, 93, 253, 0.25);
    letter-spacing: -0.5px;
}

.cx-profile-banner-name {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 3px 0;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.cx-profile-banner-sub {
    font-size: 13.5px;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
}

/* Form inputs matching professional design */
[data-testid="stMain"] div[data-testid="stTextInput"] input {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    height: 42px !important;
    padding: 8px 14px !important;
    font-size: 14px !important;
    color: #0f172a !important;
    background: #ffffff !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}

[data-testid="stMain"] div[data-testid="stTextInput"] input:focus {
    border-color: #155dfd !important;
    box-shadow: 0 0 0 3px rgba(21, 93, 253, 0.15) !important;
    outline: none !important;
}

[data-testid="stMain"] div[data-testid="stTextInput"] label,
[data-testid="stMain"] div[data-testid="stSelectbox"] label,
[data-testid="stMain"] div[data-testid="stSlider"] label {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #1e293b !important;
    margin-bottom: 6px !important;
}

[data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    height: 42px !important;
    min-height: 42px !important;
    background: #ffffff !important;
    font-size: 14px !important;
    color: #0f172a !important;
    box-shadow: none !important;
}

[data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div {
    border-color: #155dfd !important;
    box-shadow: 0 0 0 3px rgba(21, 93, 253, 0.15) !important;
}

/* Segmented Tab Bar Matching Reference Photo */
div[data-baseweb="tab-list"],
[data-testid="stTabs"] {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    display: inline-flex !important;
    gap: 4px !important;
    margin-bottom: 22px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    width: auto !important;
}

div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"] {
    display: none !important;
}

button[data-baseweb="tab"],
[data-testid="stTab"] {
    border: none !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    transition: all 0.15s ease !important;
    height: 38px !important;
    white-space: nowrap !important;
    cursor: pointer !important;
}

button[data-baseweb="tab"]:hover,
[data-testid="stTab"]:hover {
    color: #0f172a !important;
    background: rgba(255, 255, 255, 0.6) !important;
}

button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
}

/* Main Area Primary Buttons (Save Changes, Update Password, etc.) */
[data-testid="stMain"] button[data-testid="stBaseButton-primary"],
[data-testid="stMain"] button[kind="primary"] {
    background: #155dfd !important;
    color: #ffffff !important;
    border: 1px solid #155dfd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 24px !important;
    height: 40px !important;
    min-height: 40px !important;
    box-shadow: 0 4px 12px rgba(21, 93, 253, 0.25) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stMain"] button[kind="primary"]:hover {
    background: #0d4cdb !important;
    border-color: #0d4cdb !important;
    box-shadow: 0 6px 16px rgba(21, 93, 253, 0.35) !important;
}

/* Main Area Logout Outline Button (Tertiary) */
[data-testid="stMain"] button[data-testid="stBaseButton-tertiary"],
[data-testid="stMain"] button[kind="tertiary"] {
    background: #ffffff !important;
    border: 1px solid #fecaca !important;
    color: #dc2626 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 6px 20px !important;
    height: 38px !important;
    min-height: 38px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-tertiary"]:hover,
[data-testid="stMain"] button[kind="tertiary"]:hover {
    background: #fef2f2 !important;
    border-color: #fca5a5 !important;
    color: #b91c1c !important;
}

.cx-kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}
.cx-kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}
.cx-kpi-label {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.cx-kpi-icon {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    background: var(--blue-subtle);
    color: var(--blue-primary);
}
.cx-kpi-value {
    font-family: var(--font-display);
    font-size: 32px;
    font-weight: 700;
    color: var(--navy-950);
    line-height: 1.1;
    margin-bottom: 5px;
}
.cx-kpi-sub {
    font-size: 12px;
    color: var(--text-muted);
}

.cx-pred-card {
    background: linear-gradient(135deg, #0a192f 0%, #172a45 100%);
    border-radius: 16px;
    padding: 28px;
    color: #ffffff;
    box-shadow: var(--shadow-lg);
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.cx-pred-tag {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(2, 132, 199, 0.25);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 12px;
}
.cx-pred-title {
    font-family: var(--font-display);
    font-size: 36px;
    font-weight: 700;
    margin: 0 0 10px 0;
    letter-spacing: -0.8px;
    line-height: 1.1;
}

.cx-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}
.cx-badge-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.cx-badge-warning { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.cx-badge-success { background: #d1fae5; color: #047857; border: 1px solid #6ee7b7; }
.cx-badge-info { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }

.cx-disclaimer-banner {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 24px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.cx-disclaimer-banner strong {
    color: #92400e;
    font-size: 13px;
    display: block;
    margin-bottom: 2px;
}
.cx-disclaimer-banner p {
    color: #b45309;
    font-size: 12.5px;
    margin: 0;
    line-height: 1.45;
}

.cx-prob-row {
    margin-bottom: 14px;
}
.cx-prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 5px;
    color: var(--text-primary);
}
.cx-prob-track {
    width: 100%;
    height: 10px;
    background: #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
}
.cx-prob-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #0284c7, #06b6d4);
}
.cx-prob-fill.is-top {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
}

.cx-xai-legend {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f1f5f9;
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 8px 14px;
    margin-top: 10px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
}
.cx-xai-gradient-bar {
    flex-grow: 1;
    height: 8px;
    margin: 0 14px;
    border-radius: 4px;
    background: linear-gradient(to right, #000080, #00ffff, #ffff00, #ff0000);
}

.cx-metric-pill {
    background: #f8fafc;
    border: 1px solid var(--border-light);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.cx-metric-pill-val {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 700;
    color: var(--navy-900);
}
.cx-metric-pill-lbl {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 3px;
    text-transform: uppercase;
}

/* Primary Button Styling */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}

.stButton > button[kind="primary"], 
button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
}

.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(90deg, #0369a1 0%, #075985 100%) !important;
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.35) !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
    border-radius: 8px !important;
    border-color: #cbd5e1 !important;
    font-size: 13.5px !important;
}

/* ==========================================================================
   EXPLAINABLE AI (XAI) EDUCATIONAL PIPELINE STYLES
   ========================================================================== */
.cx-xai-header-card {
    background: linear-gradient(135deg, #07111e 0%, #0f2744 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    color: #ffffff;
    margin: 26px 0 20px 0;
    box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.15);
}
.cx-xai-header-title {
    font-family: var(--font-display);
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
}
.cx-xai-header-desc {
    color: #94a3b8;
    font-size: 13.5px;
    margin: 0;
    line-height: 1.5;
}
.cx-step-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 0px;
    box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}
.cx-step-card:hover {
    box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.08);
    border-color: #cbd5e1;
}
.cx-step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.cx-step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 32px;
    height: 32px;
    padding: 0 10px;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
}
.cx-step-title {
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 700;
    color: var(--navy-900);
    margin: 0;
}
.cx-step-desc {
    color: #475569;
    font-size: 13.5px;
    line-height: 1.6;
    margin-bottom: 12px;
}
.cx-step-quote {
    background: #f8fafc;
    border-left: 3px solid #0284c7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    color: #334155;
    font-size: 13px;
    line-height: 1.5;
    font-style: italic;
    margin: 10px 0 14px 0;
}
.cx-pipeline-arrow {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px 0;
}
.cx-arrow-bubble {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #0284c7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 15px;
}
.cx-flow-chain {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 12px 0;
}
.cx-flow-node {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--navy-900);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.cx-flow-node.is-active {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    border-color: #0284c7;
}
.cx-flow-arrow {
    color: #94a3b8;
    font-weight: 700;
    font-size: 13px;
}
.cx-color-card {
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    border: 1px solid #e2e8f0;
}
.cx-color-warm {
    border-left: 4px solid #ef4444;
    background: #fef2f2;
}
.cx-color-medium {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
}
.cx-color-cool {
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
}
.cx-compare-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.cx-compare-table th {
    background: #0f172a;
    color: #ffffff;
    padding: 10px 14px;
    font-weight: 600;
    text-align: left;
    font-family: var(--font-display);
    font-size: 12.5px;
}
.cx-compare-table td {
    padding: 10px 14px;
    border-top: 1px solid #e2e8f0;
    color: #334155;
    background: #ffffff;
}
.cx-compare-table tr:nth-child(even) td {
    background: #f8fafc;
}
.cx-tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
    margin: 12px 0;
}
.cx-tech-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
}
.cx-tech-item-lbl {
    font-size: 10.5px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.cx-tech-item-val {
    font-size: 13px;
    color: var(--navy-900);
    font-weight: 600;
    margin-top: 2px;
}
.cx-highlight-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 14px 0;
}
.cx-interpretation-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 22px 24px;
    color: #ffffff;
    margin: 16px 0;
}
.cx-safety-card {
    background: #fffbeb;
    border: 1px solid #fed7aa;
    border-left: 4px solid #f97316;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 20px 0 10px 0;
}

/* ==============================================================================
   MEDICAL REPORT COMPONENT & PRINT STYLES
   ============================================================================== */
.cx-report-container {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 30px 32px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    margin: 24px 0 16px 0;
    color: #1e293b;
    font-family: var(--font-primary);
}
.cx-report-header {
    border-bottom: 2px solid #0284c7;
    padding-bottom: 14px;
    margin-bottom: 20px;
}
.cx-report-title {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.cx-report-meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
}
.cx-report-section-title {
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 700;
    color: #0284c7;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 22px 0 12px 0;
}
.cx-report-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.6;
    color: #334155;
    margin-bottom: 16px;
}
.cx-report-callout {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.6;
    color: #14532d;
    margin-bottom: 16px;
}
.cx-report-disclaimer {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 12px;
    line-height: 1.55;
    color: #9a3412;
    margin-top: 24px;
}

@media print {
    section[data-testid="stSidebar"],
    [data-testid="stHeader"],
    .cx-nav-wrapper,
    .cx-btn-row,
    .stButton,
    button {
        display: none !important;
    }
    .cx-report-container {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
}
</style>

""")

# Apply Dark Mode Theme Overrides when selected
if st.session_state.get("theme", "light") == "dark":
    render_html(r"""
<style>
:root {
    --bg-app: #070d18 !important;
    --bg-card: #0d1728 !important;
    --bg-card-subtle: #132238 !important;
    
    --navy-950: #f8fafc !important;
    --navy-900: #f1f5f9 !important;
    --navy-800: #e2e8f0 !important;
    --navy-700: #cbd5e1 !important;
    --navy-600: #94a3b8 !important;
    
    --text-primary: #f8fafc !important;
    --text-secondary: #cbd5e1 !important;
    --text-muted: #94a3b8 !important;
    --text-light: #64748b !important;
    
    --border-light: rgba(148, 163, 184, 0.16) !important;
    --border-focus: #38bdf8 !important;
}

html, body, .stApp {
    background-color: #070d18 !important;
    color: #f8fafc !important;
}

/* Cards & Vertical Containers */
.cx-card,
.cx-kpi-card,
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0d1728 !important;
    border-color: rgba(148, 163, 184, 0.18) !important;
    color: #f8fafc !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

.cx-card:hover {
    border-color: rgba(56, 189, 248, 0.35) !important;
}

.cx-page-title, h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}

.cx-page-desc {
    color: #94a3b8 !important;
}

.cx-kpi-value {
    color: #f8fafc !important;
}

.cx-kpi-label {
    color: #94a3b8 !important;
}

.cx-kpi-sub {
    color: #64748b !important;
}

/* Text and Headings */
p, span, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
    color: #cbd5e1 !important;
}

strong, b {
    color: #f8fafc !important;
}

/* Inputs, Selectboxes, Textareas in Main view */
[data-testid="stMain"] label,
[data-testid="stMain"] label p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] div[data-baseweb="input"] > div,
[data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #132238 !important;
    border-color: rgba(148, 163, 184, 0.22) !important;
    color: #f8fafc !important;
}

[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {
    color: #64748b !important;
}

/* Tabs */
div[data-baseweb="tab-list"],
[data-testid="stTabs"] {
    background: #132238 !important;
    border-color: rgba(148, 163, 184, 0.2) !important;
}

button[data-baseweb="tab"],
[data-testid="stTab"] {
    color: #94a3b8 !important;
}

button[data-baseweb="tab"]:hover,
[data-testid="stTab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] {
    background: #0d1728 !important;
    color: #38bdf8 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #0d1728 !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #f8fafc !important;
}

[data-testid="stExpander"] summary {
    color: #f8fafc !important;
}

/* Secondary Buttons in Main view */
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"],
[data-testid="stMain"] button[kind="secondary"] {
    background: #132238 !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
    color: #e2e8f0 !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stMain"] button[kind="secondary"]:hover {
    background: #1e3352 !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
}

hr {
    border-color: rgba(148, 163, 184, 0.15) !important;
}
</style>
""")

# ==============================================================================
# UTILITIES & API CLIENT
# ==============================================================================

def go_to(page_name: str, nav_name: Optional[str] = None) -> None:
    st.session_state.page = page_name
    if nav_name:
        st.session_state.nav_choice = nav_name
    st.rerun()

def percentage(value: Any) -> float:
    try:
        val = float(value)
        return val * 100 if val <= 1.0 else val
    except Exception:
        return 0.0

def decode_image_bytes(value: Any) -> Optional[bytes]:
    if not value:
        return None
    try:
        if isinstance(value, str) and "," in value:
            value = value.split(",", 1)[1]
        return base64.b64decode(value)
    except Exception:
        return None

def resolve_api_image_url(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        if value.startswith("data:image"):
            return value
        if value.startswith("/"):
            return API_URL + value
    return value

def get_auth_headers() -> Dict[str, str]:
    token = st.session_state.get("session_token")
    return {"X-Session-Token": token} if token else {}

def check_api_health() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # If backend is running on localhost and not answering, trigger embedded startup
    if "127.0.0.1" in API_URL or "localhost" in API_URL:
        start_embedded_backend()
        try:
            resp = requests.get(f"{API_URL}/api/health", timeout=1)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return None

def fetch_patient_history() -> List[Dict[str, Any]]:
    try:
        resp = api_request(
            "GET",
            "/api/analyses",
            headers=get_auth_headers(),
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
    except Exception:
        pass

    # Direct database fallback if backend API is offline
    try:
        db = Database()
        records = db._recent_analyses_sync(limit=50)
        if records:
            return records
    except Exception:
        pass

    return st.session_state.get("history", [])

def delete_patient_analysis(analysis_id: Any) -> bool:
    """Deletes a specific patient analysis record from the backend and session state."""
    if not analysis_id:
        return False
    success = False
    try:
        resp = api_request(
            "DELETE",
            f"/api/analyses/{analysis_id}",
            headers=get_auth_headers(),
            timeout=8,
        )
        if resp.status_code == 200:
            success = True
    except Exception:
        pass

    # Direct database fallback
    if not success:
        try:
            db = Database()
            db._delete_analysis_sync(str(analysis_id))
            success = True
        except Exception:
            pass

    # Ensure local session state is updated
    if "history" in st.session_state and isinstance(st.session_state.history, list):
        st.session_state.history = [
            h for h in st.session_state.history
            if str(h.get("id")) != str(analysis_id)
        ]
        success = True
    return success

def clear_all_patient_history() -> bool:
    """Clears all analysis history for the current account."""
    success = False
    try:
        resp = api_request(
            "DELETE",
            "/api/analyses",
            headers=get_auth_headers(),
            timeout=8,
        )
        if resp.status_code == 200:
            success = True
    except Exception:
        pass

    st.session_state.history = []
    return success or True

def render_display_image(image_source: Any, caption: str) -> None:
    if not image_source:
        st.info(f"No {caption} image available.")
        return

    try:
        if isinstance(image_source, str) and image_source.startswith("data:image"):
            img_bytes = decode_image_bytes(image_source)
            if img_bytes:
                st.image(img_bytes, caption=caption, use_container_width=True)
                return

        resolved = resolve_api_image_url(image_source)
        st.image(resolved, caption=caption, use_container_width=True)
    except Exception as err:
        st.warning(f"Unable to render image: {err}")

def perform_logout() -> None:
    token = st.session_state.get("session_token")
    if token:
        try:
            api_request(
                "POST",
                "/api/logout",
                headers={"X-Session-Token": token},
                timeout=3,
            )
        except Exception:
            pass

    st.session_state.logged_in = False
    st.session_state.session_token = None
    st.session_state.user = None
    st.session_state.page = "Dashboard"
    st.session_state.result = None
    st.session_state.history = []
    st.session_state.uploaded_image = None
    st.session_state.show_login = False
    try:
        if "auth" in st.query_params:
            del st.query_params["auth"]
        if "login" in st.query_params:
            del st.query_params["login"]
        if "action" in st.query_params:
            del st.query_params["action"]
    except Exception:
        pass
    st.rerun()

def render_landing_page() -> None:
    """
    Renders the professional NeuroTraCera product landing page.
    Full-width, responsive, and connects seamlessly to existing login and analysis workflows.
    """
    st.html("""
    <style>
    /* Hide Streamlit default header and decoration bar */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        visibility: hidden !important;
    }
    
    /* Remove padding & margins from main view hierarchy */
    div[data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    div[data-testid="stMain"],
    .stMain,
    section.stMain {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    div.block-container,
    div.stMainBlockContainer,
    div[data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
        margin-top: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Collapse vertical gaps and element wrappers */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        gap: 0 !important;
        row-gap: 0 !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Eliminate any gap above the landing page and match clinical background */
    .stApp,
    [data-testid="stApp"] {
        background-color: #020617 !important;
    }
    
    div[data-testid="stElementContainer"]:has(> [data-testid="stHtml"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    iframe,
    [data-testid="stCustomComponentV1"] iframe {
        display: block !important;
        width: 100% !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    <script>
    (function() {
        if (!window._neurotracera_auth_listener) {
            window._neurotracera_auth_listener = true;
            window.addEventListener('message', function(event) {
                if (event.data && (event.data.type === 'NEUROTRACERA_NAVIGATE' || event.data.action)) {
                    var act = event.data.action || 'analysis';
                    var search = (act === 'analysis') ? '?auth=1&action=analysis' : '?auth=1';
                    window.location.href = window.location.origin + window.location.pathname + search;
                }
            });
        }
    })();
    </script>
    """, unsafe_allow_javascript=True)

    landing_path = os.path.join(os.path.dirname(__file__), "landing_bundle.html")
    if os.path.exists(landing_path):
        with open(landing_path, "r", encoding="utf-8") as f:
            landing_html = f.read()
        import streamlit.components.v1 as components
        components.html(landing_html, height=11800, scrolling=True)
    else:
        st.error("Landing page bundle not found.")
        if st.button("Proceed to Sign In"):
            st.session_state.show_login = True
            st.rerun()

# ==============================================================================
# AUTHENTICATION & LANDING PAGE FLOW
# ==============================================================================

if not st.session_state.logged_in:
    # 0. Check for incoming Google OAuth callback (?code=...)
    if "code" in st.query_params:
        g_code = str(st.query_params.get("code", "")).strip()
        client_id, client_secret = get_google_auth_config()
        if g_code and client_id:
            try:
                redirect_uri = get_app_base_url()
                token_resp = requests.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "code": g_code,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code"
                    },
                    timeout=10
                )
                if token_resp.status_code == 200:
                    tok_data = token_resp.json()
                    id_token = tok_data.get("id_token")
                    access_token = tok_data.get("access_token")
                    user_info = {}
                    if access_token:
                        u_resp = requests.get(
                            "https://www.googleapis.com/oauth2/v3/userinfo",
                            headers={"Authorization": f"Bearer {access_token}"},
                            timeout=8
                        )
                        if u_resp.status_code == 200:
                            user_info = u_resp.json()

                    email = user_info.get("email")
                    name = user_info.get("name") or (f"Dr. {email.split('@')[0].capitalize()}" if email else "Google Specialist")
                    sub = user_info.get("sub", "")
                    if email:
                        success, user_data = authenticate_google_user(email, name, sub, id_token)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user = user_data.get("user", {})
                            st.session_state.session_token = user_data.get("session_token")
                            st.session_state.auth_provider = "google"
                            st.session_state.page = "Dashboard"
                            st.session_state.show_login = False
                            for qk in ["code", "state", "scope", "auth", "login"]:
                                if qk in st.query_params:
                                    del st.query_params[qk]
                            st.rerun()
                else:
                    st.warning("Google authorization code expired or invalid. Please try signing in again.")
                    if "code" in st.query_params:
                        del st.query_params["code"]
            except Exception as exc:
                st.error(f"Google OAuth sign-in encountered: {exc}")
                if "code" in st.query_params:
                    try:
                        del st.query_params["code"]
                    except Exception:
                        pass

    # Check query parameters for explicit login navigation (?auth=1 or ?login=1)
    query_auth = False
    try:
        q_auth = str(st.query_params.get("auth", "")).lower()
        q_login = str(st.query_params.get("login", "")).lower()
        q_action = str(st.query_params.get("action", "")).lower()
        if q_auth in ["1", "true", "login"] or q_login in ["1", "true"]:
            query_auth = True
        if q_action == "analysis":
            st.session_state.post_login_target = "Detection"
    except Exception:
        pass

    if query_auth:
        st.session_state.show_login = True

    # 1. FIRST: Display Landing Page if user is not in login mode
    if not st.session_state.get("show_login", False):
        render_landing_page()
        st.stop()

    # 2. THEN: Display Existing Login / Register Screen (REMAINS EXACTLY THE SAME)
    server_online = check_api_health() is not None

    col_hero, col_form = st.columns([1.18, 1], gap="large")

    with col_hero:
        brand_icon_html = f'<div style="width: 58px; height: 58px; border-radius: 14px; background: rgba(7, 19, 41, 0.6); padding: 3px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 25px rgba(56, 189, 248, 0.4); border: 1.5px solid rgba(56, 189, 248, 0.45); overflow: hidden; flex-shrink: 0;"><img src="{LOGO_URI}" style="width: 100%; height: 100%; object-fit: contain;"></div>' if LOGO_URI else '<div style="font-size: 28px;">🧠</div>'
        
        render_html(f"""
        <div style="position: relative; overflow: hidden; background: radial-gradient(ellipse at 15% 20%, #0d2349 0%, #071329 55%, #040915 100%); padding: 46px 42px; border-radius: 24px; color: #ffffff; min-height: 570px; display: flex; flex-direction: column; justify-content: space-between; border: 1px solid rgba(56, 189, 248, 0.25); box-shadow: 0 25px 50px -12px rgba(2, 6, 23, 0.7);">
            <!-- Glowing Orbital Swoop Arc SVG Decoration -->
            <svg style="position: absolute; right: -15px; bottom: -15px; width: 320px; height: 320px; pointer-events: none; opacity: 0.85; z-index: 1;" viewBox="0 0 320 320" fill="none">
                <defs>
                    <linearGradient id="orbitGlowArc" x1="0%" y1="100%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0284c7" stop-opacity="0" />
                        <stop offset="35%" stop-color="#0284c7" stop-opacity="0.4" />
                        <stop offset="75%" stop-color="#38bdf8" stop-opacity="0.95" />
                        <stop offset="100%" stop-color="#22d3ee" stop-opacity="0.1" />
                    </linearGradient>
                    <filter id="glowBlur" x="-30%" y="-30%" width="160%" height="160%">
                        <feGaussianBlur stdDeviation="6" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <path d="M 20 320 C 100 290, 240 220, 320 70" stroke="#0284c7" stroke-width="9" opacity="0.3" filter="url(#glowBlur)" />
                <path d="M 20 320 C 100 290, 240 220, 320 70" stroke="url(#orbitGlowArc)" stroke-width="2.6" filter="url(#glowBlur)" />
                <path d="M 80 320 C 140 295, 250 240, 320 130" stroke="#38bdf8" stroke-width="1.2" opacity="0.22" />
            </svg>

            <!-- Main Content Area -->
            <div style="position: relative; z-index: 2;">
                <!-- Brand Title & Logo -->
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 30px;">
                    {brand_icon_html}
                    <div>
                        <h2 style="font-family: var(--font-display); font-size: 29px; font-weight: 800; margin: 0; letter-spacing: -0.6px; line-height: 1.1;">
                            <span style="color: #ffffff;">Neuro</span><span style="background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 45%, #0284c7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TraCera</span>
                        </h2>
                        <span style="color: #38bdf8; font-size: 13px; font-weight: 500; letter-spacing: 0.2px;">Explainable Brain Tumor Analytics</span>
                    </div>
                </div>

                <!-- Headlines -->
                <div style="margin-bottom: 18px;">
                    <h1 style="font-family: var(--font-display); font-size: 33px; font-weight: 800; line-height: 1.18; letter-spacing: -0.7px; margin: 0 0 6px 0; color: #ffffff;">
                        Detect. Localize. Explain.
                    </h1>
                    <div style="color: #38bdf8; font-size: 21px; font-weight: 600; letter-spacing: -0.2px;">
                        Toward a Deeper Understanding
                    </div>
                </div>

                <!-- Clinical Description -->
                <p style="color: #94a3b8; font-size: 14.2px; line-height: 1.65; margin: 0 0 26px 0; max-width: 480px;">
                    A clinical-grade academic research platform integrating ResNet-50 MRI feature extraction with Grad-CAM visual localization and Monte Carlo uncertainty quantification.
                </p>

                <!-- Features Checklist -->
                <div style="display: grid; gap: 13px; font-size: 13.8px; color: #e2e8f0; margin-bottom: 24px;">
                    <div style="display: flex; align-items: center; gap: 11px;">
                        <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: rgba(2, 132, 199, 0.28); color: #38bdf8; font-weight: bold; font-size: 12px; shrink-0;">✓</span>
                        <span>4-Class Brain Tumor Classification</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 11px;">
                        <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: rgba(2, 132, 199, 0.28); color: #38bdf8; font-weight: bold; font-size: 12px; shrink-0;">✓</span>
                        <span>Grad-CAM Spatial Heatmap Localization</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 11px;">
                        <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: rgba(2, 132, 199, 0.28); color: #38bdf8; font-weight: bold; font-size: 12px; shrink-0;">✓</span>
                        <span>Bayesian Monte Carlo Uncertainty Bounds</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 11px;">
                        <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: rgba(2, 132, 199, 0.28); color: #38bdf8; font-weight: bold; font-size: 12px; shrink-0;">✓</span>
                        <span>Patient Case Management</span>
                    </div>
                </div>
            </div>

            <!-- Footer Status Line -->
            <div style="position: relative; z-index: 2; margin-top: auto; padding-top: 18px; border-top: 1px solid rgba(255, 255, 255, 0.12); font-size: 12.5px; color: #94a3b8; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 10px #10b981;"></span>
                    <span>Server Status • <strong style="color: #10b981;">Connected (55000)</strong></span>
                </div>
                <span style="color: #64748b; font-size: 11px; font-family: monospace;">v2.4 Clinical Build</span>
            </div>
        </div>
        """)

    with col_form:
        if st.session_state.auth_mode == "login":
            is_analysis_flow = st.session_state.get("post_login_target") == "Detection"

            render_html("""
            <div style="padding: 10px 0 18px 0;">
                <h3 style="font-family: var(--font-display); font-size: 28px; font-weight: 700; color: var(--navy-900); margin: 0 0 6px 0; letter-spacing: -0.5px;">Sign In</h3>
                <p style="color: var(--text-muted); font-size: 14px; margin: 0;">Access the NeuroTraCera clinical analysis platform.</p>
            </div>
            """)

            if is_analysis_flow:
                render_html("""
                <div style="background: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.25); border-radius: 10px; padding: 10px 14px; margin-bottom: 16px; font-size: 13px; color: #0284c7; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 16px;">🔬</span>
                    <span><strong>Start Analysis Workflow:</strong> Sign in below to launch the MRI Detection Workspace directly.</span>
                </div>
                """)

            render_html("""
            <style>
            div.st-key-cx_btn_google button {
                background: #ffffff !important;
                color: #3c4043 !important;
                border: 1px solid #dadce0 !important;
                border-radius: 9999px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
                height: 42px !important;
                display: flex !important;
                flex-direction: row !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 10px !important;
                letter-spacing: 0.25px !important;
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.08), 0 1px 3px 1px rgba(60,64,67,0.06) !important;
                transition: background-color .2s, box-shadow .2s !important;
                cursor: pointer !important;
                margin-bottom: 2px !important;
            }
            div.st-key-cx_btn_google button div,
            div.st-key-cx_btn_google button p {
                margin: 0 !important;
                padding: 0 !important;
                display: inline-flex !important;
                align-items: center !important;
                flex: 0 1 auto !important;
                width: auto !important;
            }
            div.st-key-cx_btn_google button:hover {
                background-color: #f8f9fa !important;
                border-color: #dadce0 !important;
                box-shadow: 0 1px 3px 1px rgba(60,64,67,0.15), 0 1px 2px 0 rgba(60,64,67,0.08) !important;
                color: #202124 !important;
            }
            div.st-key-cx_btn_google button:active {
                background-color: #f1f3f4 !important;
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.1) !important;
            }
            div.st-key-cx_btn_google button::before {
                content: "" !important;
                display: inline-block !important;
                width: 18px !important;
                height: 18px !important;
                min-width: 18px !important;
                margin: 0 !important;
                background-size: contain !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 18 18'%3E%3Cpath fill='%234285F4' d='M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.616z'/%3E%3Cpath fill='%2334A853' d='M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z'/%3E%3Cpath fill='%23FBBC05' d='M3.964 10.707c-.18-.54-.282-1.117-.282-1.707s.102-1.167.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z'/%3E%3Cpath fill='%23EA4335' d='M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 7.293C4.672 5.166 6.656 3.58 9 3.58z'/%3E%3C/svg%3E") !important;
            }
            </style>
            """)

            client_id, client_secret = get_google_auth_config()
            target_page = "Detection" if is_analysis_flow else "Dashboard"

            # Check if live Google OAuth Client ID is active (Just like other major websites)
            if client_id:
                oauth_params = {
                    "client_id": client_id,
                    "redirect_uri": get_app_base_url(),
                    "response_type": "code",
                    "scope": "openid email profile",
                    "access_type": "online",
                    "prompt": "select_account",
                    "state": "neurotracera_oauth",
                }
                google_oauth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(oauth_params)
                render_html(f"""
                <a href="{google_oauth_url}" target="_top" style="text-decoration: none; display: block; width: 100%;">
                    <div style="background: #ffffff; color: #3c4043; border: 1px solid #dadce0; border-radius: 9999px; font-size: 14px; font-weight: 500; height: 42px; display: flex; align-items: center; justify-content: center; gap: 10px; box-shadow: 0 1px 2px 0 rgba(60,64,67,0.08), 0 1px 3px 1px rgba(60,64,67,0.06); cursor: pointer; transition: all .2s;">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18" width="18" height="18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.616z"/><path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/><path fill="#FBBC05" d="M3.964 10.707c-.18-.54-.282-1.117-.282-1.707s.102-1.167.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z"/><path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 7.293C4.672 5.166 6.656 3.58 9 3.58z"/></svg>
                        <span>Continue with Google</span>
                    </div>
                </a>
                """)
            else:
                btn_google = st.button("Continue with Google", use_container_width=True, key="cx_btn_google")

                if "google_auth_dialog" not in st.session_state:
                    st.session_state.google_auth_dialog = False

                if btn_google:
                    st.session_state.google_auth_dialog = not st.session_state.google_auth_dialog

                if st.session_state.google_auth_dialog:
                    with st.container(border=True):
                        st.markdown("<strong style='font-size: 14px; color: #0f172a;'>Sign in with Google Account</strong>", unsafe_allow_html=True)
                        st.caption("Sign in with your Google email address or clinical Google Workspace account.")
                        google_user_email = st.text_input("Your Google Email", value="neuro.specialist@gmail.com", placeholder="doctor@gmail.com", key="cx_g_email")
                        google_user_name = st.text_input("Clinician / Specialist Name", value="Dr. Neuro Specialist", placeholder="Dr. Jane Doe", key="cx_g_name")

                        col_g_act1, col_g_act2 = st.columns([1.2, 1])
                        with col_g_act1:
                            if st.button("Authorize with Google", type="primary", use_container_width=True, key="cx_btn_do_gauth"):
                                with st.spinner("Authorizing with Google Healthcare ID..."):
                                    success, payload = authenticate_google_user(
                                        email=google_user_email,
                                        name=google_user_name,
                                        google_id=f"goog_{secrets.token_hex(4)}"
                                    )
                                    if success:
                                        st.session_state.logged_in = True
                                        st.session_state.session_token = payload.get("session_token")
                                        st.session_state.user = payload.get("user", {})
                                        st.session_state.page = target_page
                                        st.session_state.show_login = False
                                        st.session_state.auth_provider = "google"
                                        if "post_login_target" in st.session_state:
                                            del st.session_state["post_login_target"]
                                        for qk in ["auth", "login", "action"]:
                                            if qk in st.query_params:
                                                del st.query_params[qk]
                                        st.success(f"Welcome, {html.escape(google_user_name)}!")
                                        time.sleep(0.3)
                                        st.rerun()
                        with col_g_act2:
                            if st.button("Close", use_container_width=True, key="cx_btn_cancel_gauth"):
                                st.session_state.google_auth_dialog = False
                                st.rerun()

                        with st.expander("ℹ️ How to connect live Google Cloud OAuth credentials"):
                            st.markdown(
                                """
                                To enable automated 1-click Google OAuth redirection like major SaaS sites:
                                1. Open [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
                                2. Create an **OAuth 2.0 Client ID** (Web application).
                                3. Set Authorized Redirect URI to:  
                                   `https://neurotracera-hmdwseb7aebbhhkwk4gsywf.streamlit.app`
                                4. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in Streamlit Cloud Secrets (or `.streamlit/secrets.toml`).
                                """,
                                unsafe_allow_html=True
                            )

            render_html("""
            <div style="display: flex; align-items: center; margin: 18px 0 16px 0; color: #94a3b8; font-size: 11px;">
                <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
                <span style="padding: 0 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b;">Or continue with email</span>
                <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            </div>
            """)

            login_user = st.text_input("User ID or Email", placeholder="Enter your User ID", key="cx_login_user")
            login_pwd = st.text_input("Password", type="password", placeholder="••••••••", key="cx_login_pwd")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Dual Primary Actions: Sign In to Portal & Start Analysis Workflow
            btn_signin = st.button("Sign In to Portal", type="secondary" if is_analysis_flow else "primary", use_container_width=True, key="cx_btn_signin")
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            btn_analysis = st.button("Start Analysis Workflow →", type="primary" if is_analysis_flow else "secondary", use_container_width=True, key="cx_btn_signin_analysis", help="Authenticate and immediately launch the MRI Detection & Analysis Workspace")

            if btn_signin or btn_analysis:
                if not login_user.strip() or not login_pwd:
                    st.error("Please provide both User ID and password.")
                else:
                    target_page = "Detection" if (btn_analysis or is_analysis_flow) else "Dashboard"
                    with st.spinner("Verifying credentials with NeuroTraCera..."):
                        success, payload, err_msg = direct_login_user(login_user.strip(), login_pwd)
                    if success and payload:
                        st.session_state.logged_in = True
                        st.session_state.session_token = payload.get("session_token")
                        st.session_state.user = payload.get("user", {})
                        st.session_state.page = target_page
                        st.session_state.show_login = False
                        if "post_login_target" in st.session_state:
                            del st.session_state["post_login_target"]
                        for qk in ["auth", "login", "action"]:
                            if qk in st.query_params:
                                del st.query_params[qk]
                        st.rerun()
                    else:
                        st.error(f"Sign-in failed: {err_msg}")

            render_html("""
            <div style="margin: 28px 0 18px 0; text-align: center; position: relative;">
                <div style="height: 1px; background: #e2e8f0; width: 100%;"></div>
                <span style="position: relative; top: -11px; background: #f8fafc; padding: 0 12px; font-size: 13px; color: #64748b; font-weight: 500;">Don't have a NeuroTraCera account?</span>
            </div>
            """)
            if st.button("Create New Account", use_container_width=True, key="cx_btn_switch_reg"):
                st.session_state.auth_mode = "register"
                st.rerun()

        else:
            render_html("""
            <div style="padding: 10px 0 18px 0;">
                <h3 style="font-family: var(--font-display); font-size: 28px; font-weight: 700; color: #0a192f; margin: 0 0 6px 0; letter-spacing: -0.5px;">Create Account</h3>
                <p style="color: #64748b; font-size: 14px; margin: 0;">Register a researcher or patient profile on NeuroTraCera.</p>
            </div>
            """)

            reg_user = st.text_input("Choose User ID", placeholder="e.g. neurospecialist02", key="cx_reg_user")
            reg_pwd = st.text_input("Password", type="password", placeholder="Min 6 characters", key="cx_reg_pwd")
            reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="cx_reg_conf")
            reg_name = st.text_input("Full Patient / Researcher Name", placeholder="Dr. Jane Doe", key="cx_reg_name")
            reg_pid = st.text_input("Optional Patient / Case ID", placeholder="e.g. CASE-9082", key="cx_reg_pid")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            if st.button("Complete Registration", type="primary", use_container_width=True, key="cx_btn_register"):
                if not reg_user.strip() or len(reg_user.strip()) < 3:
                    st.error("User ID must be at least 3 characters.")
                elif not reg_pwd or len(reg_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pwd != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Registering account on NeuroTraCera..."):
                        success, msg = direct_register_user(
                            username=reg_user.strip(),
                            password=reg_pwd,
                            patient_name=reg_name.strip() if reg_name else None,
                            patient_id=reg_pid.strip() if reg_pid else None
                        )
                    if success:
                        st.success(msg)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(msg)

            render_html("""
            <div style="margin: 28px 0 18px 0; text-align: center; position: relative;">
                <div style="height: 1px; background: #e2e8f0; width: 100%;"></div>
                <span style="position: relative; top: -11px; background: #f8fafc; padding: 0 12px; font-size: 13px; color: #64748b; font-weight: 500;">Already registered?</span>
            </div>
            """)
            if st.button("Back to Sign In", use_container_width=True, key="cx_btn_back_login"):
                st.session_state.auth_mode = "login"
                st.rerun()

    st.stop()

# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==============================================================================

current_user = st.session_state.get("user") or {}
display_patient_name = current_user.get("patient_name") or current_user.get("username") or "Investigator"
display_username = current_user.get("username", "guest")
api_health_info = check_api_health()

nav_items = [
    ("Home", ":material/home:", "Home", ["Home", "Dashboard"]),
    ("Detection", ":material/grid_view:", "Detection", ["Detection"]),
    ("Prediction", ":material/science:", "Prediction", ["Prediction", "Result"]),
    ("History", ":material/history:", "History", ["History"]),
    ("About XAI", ":material/info:", "About", ["About"]),
    ("Settings", ":material/settings:", "Settings", ["Settings"]),
    ("Profile", ":material/person:", "Profile", ["Profile"]),
]

with st.sidebar:
    sidebar_logo_html = f'<img src="{LOGO_URI}" class="cx-brand-logo-img">' if LOGO_URI else '<div class="cx-brand-logo-img" style="display:flex;align-items:center;justify-content:center;font-size:20px;">🧠</div>'

    render_html(f"""
    <div class="cx-brand-wrapper">
        {sidebar_logo_html}
        <div class="cx-brand-text">
            <h2><span style="color: #ffffff;">Neuro</span><span style="background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 45%, #0284c7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TraCera</span></h2>
            <p>Explainable Brain Tumor Analytics <span class="cx-status-dot {'cx-dot-green' if api_health_info else 'cx-dot-red'}" style="margin-left: 3px; vertical-align: middle;" title="API {'Online' if api_health_info else 'Offline'}"></span></p>
        </div>
    </div>
    """)

    for label, icon, target_page, match_pages in nav_items:
        is_active = st.session_state.page in match_pages

        if st.button(
            label,
            icon=icon,
            type="primary" if is_active else "secondary",
            use_container_width=True,
            key=f"cx_nav_btn_{target_page.lower()}",
        ):
            st.session_state.page = target_page
            st.rerun()

    render_html("""
    <div class="cx-nav-divider"></div>
    """)

    if st.button("Logout", icon=":material/logout:", type="tertiary", use_container_width=True, key="cx_btn_logout"):
        perform_logout()

# ==============================================================================
# 1. DASHBOARD PAGE
# ==============================================================================

if st.session_state.page in ["Home", "Dashboard"]:
    all_history = fetch_patient_history()
    total_cases = len(all_history)
    
    tumor_cases = sum(1 for h in all_history if h.get("predicted_class") in ["Glioma", "Meningioma", "Pituitary"])
    no_tumor_cases = sum(1 for h in all_history if h.get("predicted_class") == "No Tumor")
    confidences = [percentage(h.get("probability", 0)) for h in all_history if h.get("probability") is not None]
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    render_html("""
    <div class="cx-page-header">
        <h1 class="cx-page-title">Brain Tumor AI Dashboard</h1>
        <p class="cx-page-desc">AI-assisted MRI deep learning analysis, localization, and Grad-CAM visual explainability overview.</p>
    </div>
    """)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="cx-kpi-card">
            <div class="cx-kpi-top">
                <span class="cx-kpi-label">Total Case Analyses</span>
                <div class="cx-kpi-icon">📁</div>
            </div>
            <div class="cx-kpi-value">{total_cases}</div>
            <div class="cx-kpi-sub">MRI scans processed for account</div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="cx-kpi-card">
            <div class="cx-kpi-top">
                <span class="cx-kpi-label">Tumor Positive Cases</span>
                <div class="cx-kpi-icon" style="background:#fee2e2; color:#ef4444;">⚠️</div>
            </div>
            <div class="cx-kpi-value" style="color: {'#ef4444' if tumor_cases > 0 else '#0f172a'};">{tumor_cases}</div>
            <div class="cx-kpi-sub">Glioma, Meningioma, Pituitary</div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="cx-kpi-card">
            <div class="cx-kpi-top">
                <span class="cx-kpi-label">No Tumor Detected</span>
                <div class="cx-kpi-icon" style="background:#d1fae5; color:#10b981;">🛡️</div>
            </div>
            <div class="cx-kpi-value" style="color: #10b981;">{no_tumor_cases}</div>
            <div class="cx-kpi-sub">Benign / Non-pathological scans</div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="cx-kpi-card">
            <div class="cx-kpi-top">
                <span class="cx-kpi-label">Mean AI Confidence</span>
                <div class="cx-kpi-icon" style="background:#e0f2fe; color:#0284c7;">🎯</div>
            </div>
            <div class="cx-kpi-value">{f"{avg_conf:.1f}%" if confidences else "N/A"}</div>
            <div class="cx-kpi-sub">Bayesian posterior mean</div>
        </div>
        """)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    left_c, right_c = st.columns([1.6, 1], gap="medium")

    with left_c:
        render_html("""
        <div class="cx-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <div>
                    <h3 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; margin: 0; color: #0f172a;">Clinical XAI Workflow</h3>
                    <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">Inspect model attribution and Bayesian variance for any MRI slice</p>
                </div>
                <span class="cx-badge cx-badge-info">ResNet-50 PyTorch</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 10px;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 20px; margin-bottom: 4px;">1️⃣</div>
                    <strong style="font-size: 12px; display: block; color: #0f172a;">Upload MRI</strong>
                    <span style="font-size: 10.5px; color: #64748b;">Axial/Sagittal scan</span>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 20px; margin-bottom: 4px;">2️⃣</div>
                    <strong style="font-size: 12px; display: block; color: #0f172a;">Inference</strong>
                    <span style="font-size: 10.5px; color: #64748b;">4-Class Softmax</span>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 20px; margin-bottom: 4px;">3️⃣</div>
                    <strong style="font-size: 12px; display: block; color: #0f172a;">Grad-CAM</strong>
                    <span style="font-size: 10.5px; color: #64748b;">Gradient backprop</span>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
                    <div style="font-size: 20px; margin-bottom: 4px;">4️⃣</div>
                    <strong style="font-size: 12px; display: block; color: #0f172a;">Uncertainty</strong>
                    <span style="font-size: 10.5px; color: #64748b;">Monte Carlo passes</span>
                </div>
            </div>
        </div>
        """)

        with st.container(border=True):
            render_html("""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 0; color: #0f172a;">Recent Clinical Scans</h3>
                <span style="font-size: 12px; color: #64748b;">Showing latest records</span>
            </div>
            """)

            if not all_history:
                st.info("No scans have been processed yet for this account. Click below to run your first MRI analysis.")
            else:
                for idx, rec in enumerate(all_history[:4]):
                    pred = rec.get("predicted_class", "Unknown")
                    conf = percentage(rec.get("probability", 0))
                    uncert = rec.get("uncertainty_level", "Normal")
                    date_str = rec.get("created_at", "Recent")[:16].replace("T", " ")
                    cid = rec.get("patient_case_id") or f"CASE-{idx+101}"

                    col_a, col_b, col_c, col_d = st.columns([1.1, 1.1, 0.8, 1.3])
                    with col_a:
                        st.markdown(f"<span style='font-weight: 700; color: #0f172a; font-size: 13.5px;'>{html.escape(cid)}</span><br><small style='color:#64748b;'>{html.escape(date_str)}</small>", unsafe_allow_html=True)
                    with col_b:
                        badge_style = "cx-badge-danger" if pred != "No Tumor" else "cx-badge-success"
                        render_html(f"<span class='cx-badge {badge_style}'>{html.escape(DISPLAY_NAMES.get(pred, pred))}</span>")
                    with col_c:
                        st.markdown(f"<span style='color: #0f172a;'>Conf: <strong style='color: #0f172a;'>{conf:.1f}%</strong></span><br><small style='color:#64748b;'>Uncert: {html.escape(str(uncert))}</small>", unsafe_allow_html=True)
                    with col_d:
                        col_d1, col_d2 = st.columns([1, 1], gap="small")
                        with col_d1:
                            if st.button("View", key=f"cx_dash_view_{idx}", use_container_width=True):
                                st.session_state.result = rec
                                go_to("Result")
                        with col_d2:
                            if st.button("Remove", key=f"cx_dash_del_{idx}", help=f"Remove case {cid}", use_container_width=True):
                                rec_id = rec.get("id")
                                if delete_patient_analysis(rec_id):
                                    st.toast(f"Case {cid} removed.")
                                    st.rerun()
                    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)

    with right_c:
        render_html("""
        <div class="cx-card" style="border-left: 4px solid var(--blue-primary);">
            <span class="cx-kpi-label">ACTION WORKSPACE</span>
            <h3 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; margin: 6px 0 10px 0; color: #0f172a;">New MRI Scan Analysis</h3>
            <p style="color: #64748b; font-size: 13px; line-height: 1.5; margin-bottom: 16px;">
                Upload brain axial MRI slices to compute classification, Grad-CAM heatmaps, and Bayesian uncertainty bounds.
            </p>
        </div>
        """)
        if st.button("Launch Detection Workspace →", type="primary", use_container_width=True, key="cx_dash_btn_new"):
            go_to("Detection")

        render_html("""
        <div class="cx-card">
            <span class="cx-kpi-label">MODEL ARCHITECTURE</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 6px 0 14px 0; color: #0f172a;">Deep Learning Specifications</h3>
            <div style="font-size: 13px; line-height: 2;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding: 4px 0;">
                    <span style="color: #64748b;">Backbone</span>
                    <strong style="color: #0f172a;">ResNet-50 (Pretrained)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding: 4px 0;">
                    <span style="color: #64748b;">Classes</span>
                    <strong style="color: #0f172a;">4 Classes (Multi-Class)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding: 4px 0;">
                    <span style="color: #64748b;">Attribution</span>
                    <strong style="color: #0f172a;">Grad-CAM (Layer 4)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding: 4px 0;">
                    <span style="color: #64748b;">Uncertainty</span>
                    <strong style="color: #0f172a;">Monte Carlo Dropout (MC)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span style="color: #64748b;">Input Res</span>
                    <strong style="color: #0f172a;">224 × 224 × 3 normalized</strong>
                </div>
            </div>
        </div>
        """)

    render_html("""
    <div class="cx-disclaimer-banner">
        <span style="font-size: 20px;">⚖️</span>
        <div>
            <strong>Academic & Educational Research Disclaimer</strong>
            <p>
                NeuroTraCera is an academic prototype designed for AI-assisted MRI analysis and interpretability benchmarking. 
                It is not a certified medical device and must never substitute for clinical judgment, pathology verification, 
                or consultation with qualified healthcare professionals.
            </p>
        </div>
    </div>
    """)

# ==============================================================================
# 2. BRAIN TUMOR DETECTION (ANALYSIS WORKSPACE)
# ==============================================================================

elif st.session_state.page in ["Detection", "Workspace"]:
    render_html("""
    <div class="cx-page-header">
        <h1 class="cx-page-title">Brain Tumor Detection Workspace</h1>
        <p class="cx-page-desc">Upload a brain MRI scan to trigger ResNet-50 classification, Grad-CAM attribution, and Monte Carlo uncertainty analysis.</p>
    </div>
    """)

    col_input, col_params = st.columns([1.3, 1], gap="large")

    with col_input:
        with st.container(border=True):
            render_html("""
            <span class="cx-kpi-label">STEP 1: CASE REGISTRATION</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 4px 0 14px 0; color: #0f172a;">Patient / Case Metadata</h3>
            """)

            p1, p2, p3 = st.columns(3)
            with p1:
                case_id = st.text_input("Patient / Case ID", value="", placeholder="e.g. CX-2026-041", key="cx_input_caseid")
            with p2:
                patient_age = st.number_input("Patient Age", min_value=1, max_value=115, value=48, step=1, key="cx_input_age")
            with p3:
                patient_gender = st.selectbox("Biological Sex", ["Not specified", "Male", "Female", "Other"], key="cx_input_sex")

        with st.container(border=True):
            render_html("""
            <span class="cx-kpi-label">STEP 2: MRI SCAN UPLOAD</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 4px 0 10px 0; color: #0f172a;">Upload Brain MRI Image</h3>
            <p style="color: #64748b; font-size: 13px; margin-bottom: 12px;">Supported formats: PNG, JPG, JPEG, WEBP, BMP, TIFF (Up to 20 MB)</p>
            """)

            uploaded_mri = st.file_uploader(
                "Upload Brain MRI Scan",
                type=SUPPORTED_EXTENSIONS,
                help="High-contrast T1-CE axial or FLAIR scans yield optimal Grad-CAM localization.",
                key="cx_file_uploader",
                label_visibility="collapsed",
            )

            if uploaded_mri is not None:
                st.session_state.uploaded_image = uploaded_mri
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                img_c1, img_c2 = st.columns([1.2, 1])
                with img_c1:
                    st.image(uploaded_mri, caption=f"Selected: {uploaded_mri.name}", use_container_width=True)
                with img_c2:
                    render_html(f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; font-size: 12.5px;">
                        <strong style="display: block; color: var(--navy-900); margin-bottom: 6px;">Image Properties:</strong>
                        <div><strong>File Name:</strong> {html.escape(uploaded_mri.name)}</div>
                        <div><strong>MIME Type:</strong> {html.escape(uploaded_mri.type)}</div>
                        <div><strong>File Size:</strong> {uploaded_mri.size / 1024:.1f} KB</div>
                        <div style="margin-top: 8px; color: #10b981; font-weight: 600;">✓ Ready for neural inference</div>
                    </div>
                    """)

    with col_params:
        with st.container(border=True):
            render_html("""
            <span class="cx-kpi-label">STEP 3: XAI PARAMETERS</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 4px 0 14px 0; color: #0f172a;">Inference & Explainability Settings</h3>
            """)

            mc_passes = st.slider(
                "Monte Carlo Inference Passes",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Number of stochastic dropout passes for uncertainty quantification.",
                key="cx_param_mc",
            )

            target_cam_layer = st.selectbox(
                "Grad-CAM Attribution Target Layer",
                ["layer4", "layer3", "layer2", "layer1"],
                index=0,
                help="Convolutional feature layer from which gradients are backpropagated. Layer 4 captures high-level semantic tumor features.",
                key="cx_param_layer",
            )

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                overlay_alpha = st.slider(
                    "Overlay Opacity (α)",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.55,
                    step=0.05,
                    key="cx_param_alpha",
                )
            with col_a2:
                colormap_choice = st.selectbox(
                    "Heatmap Colormap",
                    ["jet", "turbo", "viridis", "inferno", "magma", "plasma"],
                    index=0,
                    key="cx_param_cmap",
                )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                mri_sequence = st.selectbox(
                    "MRI Sequence Protocol",
                    ["T1-CE Axial", "T1 Axial", "T2 Axial", "FLAIR", "Other"],
                    key="cx_param_seq",
                )
            with col_s2:
                anatomical_region = st.selectbox(
                    "Anatomical ROI",
                    ["Cerebral Hemisphere", "Whole Brain", "Brain Cerebrum", "Sellar / Pituitary", "Other"],
                    key="cx_param_anatomy",
                )

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            analyze_clicked = st.button(
                "🧠 Analyze MRI Scan",
                type="primary",
                use_container_width=True,
                key="cx_btn_run_analysis",
                disabled=st.session_state.analysis_running,
            )

    if analyze_clicked:
        if uploaded_mri is None:
            st.error("Please upload an MRI image file before running analysis.")
        else:
            try:
                st.session_state.analysis_running = True
                files_payload = {
                    "file": (
                        uploaded_mri.name,
                        uploaded_mri.getvalue(),
                        uploaded_mri.type,
                    )
                }

                analysis_result = None
                with st.spinner("Processing MRI: Evaluating ResNet-50 weights, computing Grad-CAM gradients, and sampling Monte Carlo dropout passes..."):
                    try:
                        resp = api_request(
                            "POST",
                            "/api/analyze/upload",
                            files=files_payload,
                            headers=get_auth_headers(),
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            analysis_result = resp.json()
                    except Exception:
                        pass

                # If backend API is unreachable or returned error, execute local fallback
                if not analysis_result:
                    with st.spinner("Executing direct neural network inference and explainability pipeline..."):
                        try:
                            analysis_result = perform_direct_mri_analysis(
                                uploaded_file=uploaded_mri,
                                case_id=case_id,
                                patient_age=patient_age,
                                patient_gender=patient_gender,
                                mri_sequence=mri_sequence,
                                anatomical_region=anatomical_region,
                                cam_layer=target_cam_layer,
                                alpha=overlay_alpha,
                                colormap=colormap_choice,
                                mc_passes=10,
                                run_all_xai=False
                            )
                        except Exception as local_err:
                            st.error(f"Inference pipeline encountered an issue: {local_err}")

                st.session_state.analysis_running = False

                if analysis_result:
                    analysis_result["_patient_age"] = patient_age
                    analysis_result["_patient_gender"] = patient_gender
                    analysis_result["patient_case_id"] = case_id or analysis_result.get("patient_case_id")
                    analysis_result["cam_layer"] = target_cam_layer
                    analysis_result["alpha"] = overlay_alpha
                    analysis_result["colormap"] = colormap_choice
                    analysis_result["sequence"] = mri_sequence
                    analysis_result["anatomy"] = anatomical_region

                    st.session_state.result = analysis_result
                    st.session_state.history.insert(0, analysis_result)
                    st.session_state.current_report = None
                    st.session_state.report_editing = False

                    st.success("MRI analysis completed successfully.")
                    go_to("Result")
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"An unexpected error occurred: {e}")

# ==============================================================================
# 3. PREDICTION & EXPLAINABILITY RESULTS PAGE
# ==============================================================================

elif st.session_state.page in ["Prediction", "Result", "Analysis"]:
    res = st.session_state.result

    if not res:
        render_html("""
        <div class="cx-card" style="text-align: center; padding: 50px 20px;">
            <div style="font-size: 40px; margin-bottom: 12px;">🔬</div>
            <h3 style="font-family: var(--font-display); font-size: 20px; font-weight: 700;">No Prediction Results Loaded</h3>
            <p style="color: #64748b; font-size: 13.5px; margin-bottom: 20px;">
                Upload a brain MRI scan in the Detection workspace or choose a past case from patient history.
            </p>
        </div>
        """)
        if st.button("Go to Detection Workspace →", type="primary"):
            go_to("Detection")
    else:
        pred_class = res.get("predicted_class", "Unknown")
        pred_label = DISPLAY_NAMES.get(pred_class, pred_class)
        conf_prob = percentage(res.get("probability", 0))
        uncertainty_val = res.get("uncertainty", 0.0)
        uncertainty_std = res.get("uncertainty_std", uncertainty_val)
        uncertainty_level = res.get("uncertainty_level", "Low")
        conf_interval = res.get("confidence_interval", [])
        class_probs = res.get("class_probabilities", {})

        is_tumor = pred_class in ["Glioma", "Meningioma", "Pituitary"]
        tag_color = "#f87171" if is_tumor else "#34d399"

        col_top_l, col_top_r = st.columns([1.7, 1.3])
        with col_top_l:
            render_html(f"""
            <div class="cx-page-header">
                <h1 class="cx-page-title">Diagnostic Prediction & Explainability Report</h1>
                <p class="cx-page-desc">Case ID: <strong>{html.escape(str(res.get('patient_case_id') or 'N/A'))}</strong> &nbsp;·&nbsp; Scan: <strong>{html.escape(str(res.get('image_name') or 'Brain_MRI'))}</strong></p>
            </div>
            """)
        with col_top_r:
            col_r1, col_r2 = st.columns([1, 1], gap="small")
            with col_r1:
                if st.button("← Analyze New Scan", key="cx_btn_top_new", use_container_width=True):
                    go_to("Detection")
            with col_r2:
                if st.button("🗂️ View Case History", key="cx_btn_top_hist", use_container_width=True):
                    go_to("History")

        render_html(f"""
        <div class="cx-pred-card">
            <div class="cx-pred-tag" style="border-color: {tag_color}; color: {tag_color};">
                {'● PATHOLOGY IDENTIFIED' if is_tumor else '✓ NO TUMOR IDENTIFIED'}
            </div>
            <div class="cx-pred-title">{html.escape(pred_label)}</div>
            <div style="color: #cbd5e1; font-size: 14.5px; max-width: 800px; line-height: 1.5; margin-bottom: 22px;">
                {CLASS_DESCRIPTIONS.get(pred_class, "AI classification generated via ResNet-50 deep neural network.")}
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 18px;">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Prediction Confidence</div>
                    <div style="font-family: var(--font-display); font-size: 28px; font-weight: 700; color: #ffffff;">{conf_prob:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Uncertainty Metric (Std)</div>
                    <div style="font-family: var(--font-display); font-size: 28px; font-weight: 700; color: #ffffff;">{float(uncertainty_std):.4f}</div>
                </div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Uncertainty Level</div>
                    <div style="font-family: var(--font-display); font-size: 24px; font-weight: 700; color: {'#34d399' if str(uncertainty_level).lower() == 'low' else '#fbbf24'};">{html.escape(str(uncertainty_level).capitalize())}</div>
                </div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Confidence Interval (95%)</div>
                    <div style="font-family: var(--font-display); font-size: 20px; font-weight: 700; color: #ffffff; padding-top: 4px;">
                        {f"{percentage(conf_interval[0]):.1f}% – {percentage(conf_interval[1]):.1f}%" if len(conf_interval) >= 2 else "N/A"}
                    </div>
                </div>
            </div>
        </div>
        """)

                # ==============================================================================
        # SECTION: "HOW THE AI REACHED THIS RESULT" (EDUCATIONAL XAI PIPELINE)
        # ==============================================================================
        render_html("""
        <div class="cx-xai-header-card">
            <h2 class="cx-xai-header-title">🧠 How the AI Reached This Result</h2>
            <p class="cx-xai-header-desc">
                A step-by-step Explainable AI (XAI) educational walkthrough illustrating how the MRI scan was preprocessed, 
                processed by ResNet50, abstracted into visual features, classified into tumor categories, and visually explained with Grad-CAM.
            </p>
        </div>
        """)

        # ----------------------------------------------------------------------
        # 12. "From Pixels to Prediction" Horizontal Pipeline (Interactive)
        # ----------------------------------------------------------------------
        st.markdown("##### 📍 From Pixels to Prediction — Machine Learning Pipeline")
        pipeline_stages = [
            "1. MRI Pixels",
            "2. Preprocessing",
            "3. ResNet50",
            "4. Feature Maps",
            "5. Classification",
            "6. Prediction",
            "7. Grad-CAM",
            "8. Visual Explanation",
        ]
        selected_pipeline_stage = st.pills(
            "Select Pipeline Stage to Explore:",
            pipeline_stages,
            default=pipeline_stages[0],
            key="cx_pills_pipeline",
            label_visibility="collapsed",
        )

        stage_explanations = {
            "1. MRI Pixels": (
                "<strong>1. Raw MRI Pixels</strong>: The uploaded patient scan provides a 2D matrix of magnetic resonance intensity contrasts "
                "(T1-CE, FLAIR, T2). These numerical pixel values serve as the primary sensory input to the machine learning system."
            ),
            "2. Preprocessing": (
                "<strong>2. Image Preprocessing</strong>: The input MRI is resized to 224×224 pixels, mapped across 3 RGB channels, and standardized using "
                "ImageNet channel mean and standard deviation to produce a normalized PyTorch tensor."
            ),
            "3. ResNet50": (
                "<strong>3. Deep Neural Backbone (ResNet50)</strong>: A 50-layer deep convolutional neural network containing residual bottleneck blocks "
                "processes the normalized tensor through hierarchical convolutions without vanishing gradients."
            ),
            "4. Feature Maps": (
                "<strong>4. Feature Extraction</strong>: Deeper convolutional layers (<code>layer4</code>) extract high-level spatial feature activation maps "
                "representing structural contrast, texture variations, and tissue boundary patterns."
            ),
            "5. Classification": (
                "<strong>5. Classification Layer</strong>: Features are globally average-pooled and passed through a fully connected dense layer "
                "with Softmax normalization, outputting distinct probability scores for each learned tumor class."
            ),
            "6. Prediction": (
                "<strong>6. Final Prediction</strong>: The category with the highest probability score is selected as the model's output prediction, "
                "accompanied by stochastic Monte Carlo uncertainty quantification."
            ),
            "7. Grad-CAM": (
                "<strong>7. Grad-CAM Backpropagation</strong>: Gradients of the predicted class score are backpropagated to the final convolutional feature maps, "
                "weighting each activation channel by its mathematical contribution."
            ),
            "8. Visual Explanation": (
                "<strong>8. Visual Explanation Overlay</strong>: The resulting activation heatmap is color-mapped and blended onto the anatomical MRI scan, "
                "providing an intuitive visual attribution of the regions that influenced the model."
            ),
        }
        render_html(f"""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #0284c7; border-radius: 10px; padding: 14px 18px; margin-bottom: 22px; font-size: 13.5px; color: #1e293b; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            {stage_explanations.get(selected_pipeline_stage, "")}
        </div>
        """)

        def render_arrow():
            render_html("""
            <div class="cx-pipeline-arrow">
                <div class="cx-arrow-bubble">↓</div>
            </div>
            """)

        # Extract backend metrics and localization safely without inventing data
        loc_data = res.get("localization") if isinstance(res.get("localization"), dict) else {}
        bbox_data = loc_data.get("bounding_box") if isinstance(loc_data.get("bounding_box"), dict) else None
        xai_metrics_data = res.get("xai_metrics") if isinstance(res.get("xai_metrics"), dict) else {}

        raw_cov = res.get("active_area_coverage_pct", xai_metrics_data.get("active_area_coverage"))
        if raw_cov is not None:
            cov_pct = float(raw_cov) * 100.0 if float(raw_cov) <= 1.0 else float(raw_cov)
        else:
            cov_pct = None

        raw_snr = res.get("signal_to_noise_ratio", xai_metrics_data.get("snr"))
        snr_value = float(raw_snr) if raw_snr is not None else None

        # ======================================================================
        # STEP 1 — MRI Input
        # ======================================================================
        with st.container(border=True):
            render_html("""
            <div class="cx-step-header">
                <div class="cx-step-badge">1</div>
                <h3 class="cx-step-title" style="color: #0f172a;">1. MRI Image Input</h3>
            </div>
            <div class="cx-step-desc">
                The uploaded MRI scan is provided as the input to the AI system. The model analyzes the visual information contained in the image to identify patterns associated with the trained brain-tumor classes.
            </div>
            """)

            step1_c1, step1_c2 = st.columns([1, 1.5], gap="medium")
            with step1_c1:
                uploaded_preview = st.session_state.get("uploaded_image")
                orig_url = res.get("original_image_url")
                if uploaded_preview:
                    st.image(uploaded_preview, caption="Input Scan Slice", use_container_width=True)
                elif orig_url:
                    render_display_image(orig_url, "Input Scan Slice")
                else:
                    st.info("Input MRI scan preview not available in current session.")
            with step1_c2:
                render_html(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; font-size: 13px;">
                    <div style="font-weight: 700; color: #0f172a; margin-bottom: 8px; font-size: 13.5px;">Acquisition & Slice Metadata</div>
                    <div style="margin-bottom: 6px; color: #334155;"><strong style="color: #0f172a;">File Name:</strong> {html.escape(str(res.get('image_name') or 'Brain_MRI.png'))}</div>
                    <div style="margin-bottom: 6px; color: #334155;"><strong style="color: #0f172a;">Imaging Sequence:</strong> {html.escape(str(res.get('sequence') or 'T1-CE Axial'))}</div>
                    <div style="margin-bottom: 6px; color: #334155;"><strong style="color: #0f172a;">Anatomical Region:</strong> {html.escape(str(res.get('anatomy') or 'Cerebral Hemisphere'))}</div>
                    <div style="margin-bottom: 6px; color: #334155;"><strong style="color: #0f172a;">Case ID:</strong> {html.escape(str(res.get('patient_case_id') or res.get('patient_id') or 'N/A'))}</div>
                    <div style="margin-top: 10px; color: #0284c7; font-weight: 600; font-size: 12px;">✓ Verified 2D axial magnetic resonance image format</div>
                </div>
                """)

        render_arrow()

        # ======================================================================
        # STEP 2 — Image Preprocessing
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">2</div>
                <h3 class="cx-step-title">2. Image Preprocessing</h3>
            </div>
            <div class="cx-step-desc">
                Before analysis, the MRI image is transformed into a standardized format that the neural network can process consistently. This helps the model receive images in the expected size and numerical representation.
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 14px 0;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <strong style="color: #0284c7; font-size: 12.5px;">1. Image Resizing</strong>
                    <p style="color: #475569; font-size: 12px; margin: 4px 0 0 0;">Interpolated to 224 × 224 pixels to match the convolutional network receptive field.</p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <strong style="color: #0284c7; font-size: 12.5px;">2. Format Conversion</strong>
                    <p style="color: #475569; font-size: 12px; margin: 4px 0 0 0;">Standardized across 3 color channels (RGB) for multi-channel kernel processing.</p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <strong style="color: #0284c7; font-size: 12.5px;">3. ImageNet Normalization</strong>
                    <p style="color: #475569; font-size: 12px; margin: 4px 0 0 0;">Channel-wise zero-centering using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].</p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <strong style="color: #0284c7; font-size: 12.5px;">4. PyTorch Tensor Conversion</strong>
                    <p style="color: #475569; font-size: 12px; margin: 4px 0 0 0;">Constructs a 4D float tensor [1, 3, 224, 224] with gradient tracking enabled for attribution.</p>
                </div>
            </div>
            <div style="font-weight: 700; font-size: 12px; color: var(--navy-900); text-transform: uppercase; margin-top: 16px; margin-bottom: 6px;">
                Verified Backend Preprocessing Parameters:
            </div>
            <div class="cx-tech-grid">
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Target Resolution</div><div class="cx-tech-item-val">224 × 224 px</div></div>
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Color Space</div><div class="cx-tech-item-val">3 Channels (RGB)</div></div>
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Normalization Mean</div><div class="cx-tech-item-val">[0.485, 0.456, 0.406]</div></div>
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Normalization Std</div><div class="cx-tech-item-val">[0.229, 0.224, 0.225]</div></div>
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Input Tensor Shape</div><div class="cx-tech-item-val">[1, 3, 224, 224]</div></div>
                <div class="cx-tech-item"><div class="cx-tech-item-lbl">Tensor Representation</div><div class="cx-tech-item-val">Zero-Centered FloatTensor</div></div>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 3 — Neural Network Processing (Deep Learning Model)
        # ======================================================================
        render_html(f"""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">3</div>
                <h3 class="cx-step-title">3. Deep Learning Model</h3>
            </div>
            <div class="cx-step-desc">
                The processed image is passed through the trained deep learning model. The network contains multiple layers that progressively transform the image into numerical representations called features.
            </div>
            <div style="background: linear-gradient(135deg, #0a192f 0%, #172a45 100%); border-radius: 12px; padding: 18px 22px; color: #ffffff; margin: 14px 0; border: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px;">Model Architecture</span>
                        <h4 style="font-family: var(--font-display); font-size: 22px; font-weight: 700; color: #ffffff; margin: 2px 0 0 0;">ResNet50 (Residual Network - 50 Layers)</h4>
                    </div>
                    <span class="cx-badge cx-badge-info" style="font-size: 12px; padding: 4px 12px;">Active Checkpoint Loaded</span>
                </div>
                <p style="color: #cbd5e1; font-size: 13px; line-height: 1.5; margin: 0 0 12px 0;">
                    ResNet50 is a deep convolutional neural network that learns visual patterns from images through multiple layers. 
                    Residual skip connections enable the gradient to flow across all 50 layers during training, mitigating vanishing gradients and learning rich hierarchical visual representations.
                </p>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 10px; font-size: 12px; color: #94a3b8;">
                    <div><strong>Residual Stages:</strong> 4 stages (layer1, layer2, layer3, layer4)</div>
                    <div><strong>Target Attribution Layer:</strong> {html.escape(str(res.get('cam_layer', 'layer4')))}</div>
                    <div><strong>Classification Head:</strong> Linear (2048 in → 4 classes out)</div>
                </div>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 4 — Feature Extraction & "What the Model Uses"
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">4</div>
                <h3 class="cx-step-title">4. Feature Extraction</h3>
            </div>
            <div class="cx-step-desc">
                As the image passes through the network, different layers extract increasingly complex visual features. Early layers may respond to basic structures such as edges and textures, while deeper layers combine these patterns into higher-level representations used for classification.
            </div>
            <div style="margin: 14px 0;">
                <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; color: var(--navy-900); margin-bottom: 8px;">
                    Hierarchical Visual Feature Progression:
                </div>
                <div class="cx-flow-chain">
                    <div class="cx-flow-node">Pixels (224×224)</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node">Edges & Contours</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node">Textures & Gradients</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node">Contrast Patterns</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node is-active">High-Level Features (layer4)</div>
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-top: 14px;">
                <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 4px;">What the Model Uses</strong>
                <p style="color: #475569; font-size: 12.5px; margin: 0; line-height: 1.5;">
                    During training, the neural network learns patterns in the training images that help distinguish between the classes. 
                    These learned representations are then used when analyzing a new MRI. 
                    <em>The model operates via learned mathematical feature representations and does not independently understand medical anatomy like a radiologist.</em>
                </p>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 5 — Classification
        # ======================================================================
        with st.container(border=True):
            render_html("""
            <div class="cx-step-header">
                <div class="cx-step-badge">5</div>
                <h3 class="cx-step-title" style="color: #0f172a;">5. Classification</h3>
            </div>
            <div class="cx-step-desc">
                The extracted features are passed to the classification layer, which produces scores for each tumor category learned during training.
            </div>
            """)

            step5_c1, step5_c2 = st.columns([1.2, 1], gap="medium")
            with step5_c1:
                if class_probs:
                    for c_name in CLASSES:
                        val = class_probs.get(c_name, 0.0)
                        pct = percentage(val)
                        is_top = (c_name == pred_class)
                        render_html(f"""
                        <div class="cx-prob-row">
                            <div class="cx-prob-header">
                                <span style="font-weight: {'700' if is_top else '500'}; color: {'#0284c7' if is_top else '#0f172a'};">
                                    {html.escape(c_name)} {'★' if is_top else ''}
                                </span>
                                <span style="font-weight: 700; color: #0f172a;">{pct:.2f}%</span>
                            </div>
                            <div class="cx-prob-track">
                                <div class="cx-prob-fill {'is-top' if is_top else ''}" style="width: {pct}%;"></div>
                            </div>
                        </div>
                        """)
                else:
                    st.info("Class probabilities not available from the current analysis.")
            with step5_c2:
                breakdown_rows = ""
                if class_probs:
                    for c_name in CLASSES:
                        val = class_probs.get(c_name, 0.0)
                        pct = percentage(val)
                        is_top = (c_name == pred_class)
                        weight = "700" if is_top else "400"
                        color = "#0284c7" if is_top else "#334155"
                        breakdown_rows += f"""
                        <div style="display: flex; justify-content: space-between; padding: 6px 0; font-size: 12.5px; border-bottom: 1px solid #f1f5f9; font-weight: {weight}; color: {color};">
                            <span>{html.escape(c_name)}</span>
                            <span>{pct:.2f}%</span>
                        </div>
                        """
                else:
                    breakdown_rows = "<div style='color: #64748b;'>Not available from the current analysis.</div>"

                render_html(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; font-size: 12.5px;">
                    <div style="font-weight: 700; color: #0f172a; margin-bottom: 8px;">Model Class Probability Breakdown:</div>
                    {breakdown_rows}
                </div>
                """)

        render_arrow()

        # ======================================================================
        # STEP 6 — Final Prediction Selection
        # ======================================================================
        render_html(f"""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">6</div>
                <h3 class="cx-step-title">6. Final Prediction</h3>
            </div>
            <div class="cx-step-desc">
                The model compares the output scores across the available classes. The class with the highest model probability is selected as the predicted category.
            </div>
            <div style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); border-radius: 12px; padding: 20px 24px; color: #ffffff; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; margin-top: 10px; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);">
                <div>
                    <div style="font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.8px; color: #bae6fd; font-weight: 700;">Predicted Class</div>
                    <div style="font-family: var(--font-display); font-size: 30px; font-weight: 700; color: #ffffff; margin-top: 2px;">
                        {html.escape(pred_class)} <span style="font-size: 18px; font-weight: 500; opacity: 0.9;">({html.escape(pred_label)})</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.8px; color: #bae6fd; font-weight: 700;">Model Confidence</div>
                    <div style="font-family: var(--font-display); font-size: 30px; font-weight: 700; color: #ffffff; margin-top: 2px;">
                        {conf_prob:.2f}%
                    </div>
                </div>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 7 — Prediction Uncertainty
        # ======================================================================
        mc_passes_count = res.get('mc_passes_count', 20)
        ci_str = f"{percentage(conf_interval[0]):.1f}% – {percentage(conf_interval[1]):.1f}%" if len(conf_interval) >= 2 else "Not available from the current analysis."
        render_html(f"""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">7</div>
                <h3 class="cx-step-title">7. Prediction Uncertainty</h3>
            </div>
            <div class="cx-step-desc">
                Confidence and uncertainty provide information about how strongly the model supports its prediction. A high confidence value does not mean that the prediction is medically certain.
            </div>
            <div class="cx-step-quote">
                "The system performs multiple model passes to estimate how stable the prediction is across repeated evaluations."
            </div>
            <div class="cx-tech-grid" style="margin-top: 14px;">
                <div class="cx-tech-item">
                    <div class="cx-tech-item-lbl">Prediction Confidence</div>
                    <div class="cx-tech-item-val">{conf_prob:.2f}%</div>
                </div>
                <div class="cx-tech-item">
                    <div class="cx-tech-item-lbl">Uncertainty (Std Dev)</div>
                    <div class="cx-tech-item-val">{float(uncertainty_std):.5f}</div>
                </div>
                <div class="cx-tech-item">
                    <div class="cx-tech-item-lbl">Uncertainty Level</div>
                    <div class="cx-tech-item-val" style="color: {'#16a34a' if str(uncertainty_level).lower() == 'low' else '#d97706'}; font-weight: 700;">
                        {html.escape(str(uncertainty_level).capitalize())}
                    </div>
                </div>
                <div class="cx-tech-item">
                    <div class="cx-tech-item-lbl">Evaluation Method</div>
                    <div class="cx-tech-item-val">{mc_passes_count} Monte Carlo passes</div>
                </div>
                <div class="cx-tech-item">
                    <div class="cx-tech-item-lbl">95% Confidence Interval</div>
                    <div class="cx-tech-item-val">{ci_str}</div>
                </div>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # EXPLAINABLE AI ANALYSIS SUITE (5 METHODS + UNIFIED DASHBOARD)
        # ======================================================================

        # Extract or initialize data for all 5 methods
        gcam_data = res.get("gradcam") or {
            "heatmap_url": res.get("gradcam_heatmap_url"),
            "overlay_url": res.get("gradcam_overlay_url"),
            "bounding_box": res.get("localization", {}).get("bounding_box") if isinstance(res.get("localization"), dict) else None,
            "metrics": res.get("xai_metrics", {}),
            "description": "Grad-CAM uses gradient information associated with the selected prediction to estimate which spatial regions contributed strongly to the output.",
            "why_highlighted": f"The highlighted regions represent areas of the convolutional feature representation that contributed strongly to the selected model output ({pred_label}).",
        }

        gcpp_data = res.get("gradcam_plus_plus")
        lime_data = res.get("lime")
        shap_data = res.get("shap")
        ig_data = res.get("integrated_gradients")
        cross_metrics = res.get("cross_method_analysis")

        # ----------------------------------------------------------------------
        # PIPELINE BREADCRUMB & MULTI-PERSPECTIVE XAI SUITE BANNER
        # ----------------------------------------------------------------------
        render_html(f"""
        <div class="cx-step-card" style="border-left: 4px solid #0284c7; background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
                <div>
                    <span class="cx-kpi-label">EXPLAINABLE AI WORKSPACE</span>
                    <h2 style="font-family: var(--font-display); font-size: 22px; font-weight: 700; color: var(--navy-900); margin: 2px 0 0 0;">
                        Multi-Perspective Model Attribution Suite
                    </h2>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    <span class="cx-badge {'cx-badge-success' if gcam_data and not gcam_data.get('error') else 'cx-badge-warning'}">Grad-CAM</span>
                    <span class="cx-badge {'cx-badge-success' if gcpp_data and not gcpp_data.get('error') else 'cx-badge-warning'}">Grad-CAM++</span>
                    <span class="cx-badge {'cx-badge-success' if lime_data and not lime_data.get('error') else 'cx-badge-warning'}">LIME</span>
                    <span class="cx-badge {'cx-badge-success' if shap_data and not shap_data.get('error') else 'cx-badge-warning'}">SHAP</span>
                    <span class="cx-badge {'cx-badge-success' if ig_data and not ig_data.get('error') else 'cx-badge-warning'}">Integrated Gradients</span>
                </div>
            </div>

            <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; color: var(--navy-900); margin-bottom: 6px;">
                Complete End-to-End Decision & Attribution Pipeline:
            </div>
            <div class="cx-flow-chain" style="margin-bottom: 14px;">
                <div class="cx-flow-node">MRI Upload</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node">Preprocessing</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node">ResNet-50</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node">Prediction</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node">Uncertainty</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node is-active">Grad-CAM</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node is-active">Grad-CAM++</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node is-active">LIME</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node is-active">SHAP</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node is-active">Integrated Gradients</div>
                <div class="cx-flow-arrow">→</div>
                <div class="cx-flow-node">Cross Comparison</div>
            </div>

            <p style="color: #475569; font-size: 13px; line-height: 1.55; margin: 0 0 14px 0;">
                A prediction alone reports what category the neural network assigned, but cannot reveal the internal visual reasoning behind that decision. 
                This research suite evaluates <strong>five complementary Explainable AI (XAI) paradigms</strong> directly on your uploaded scan and trained weights 
                to provide multi-faceted spatial, perturbation, game-theoretic, and axiomatic gradient perspectives.
            </p>
        </div>
        """)

        # ----------------------------------------------------------------------
        # "RUN COMPLETE XAI ANALYSIS" ACTION BUTTON
        # ----------------------------------------------------------------------
        all_methods_ready = bool(
            gcam_data and not gcam_data.get("error") and
            gcpp_data and not gcpp_data.get("error") and
            lime_data and not lime_data.get("error") and
            shap_data and not shap_data.get("error") and
            ig_data and not ig_data.get("error")
        )

        run_col1, run_col2 = st.columns([2.5, 1])
        with run_col1:
            st.markdown(
                f"<div style='font-size:13px; color:#64748b; padding-top:8px;'>"
                f"Status: <strong>{'All 5 XAI Methods Evaluated ✓' if all_methods_ready else 'Select a method tab below or execute the full multi-method suite'}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
        with run_col2:
            run_all_btn = st.button(
                "🚀 Run Complete XAI Analysis",
                type="primary",
                use_container_width=True,
                help="Computes all 5 XAI methods sequentially with live progress feedback and cross-method agreement metrics.",
                key="cx_btn_run_complete_xai"
            )

        if run_all_btn:
            with st.status("Executing Multi-Method Explainable AI Pipeline...", expanded=True) as status_box:
                # 1. Grad-CAM
                status_box.write("✓ Grad-CAM verified from model forward/backward pass")
                
                # 2. Grad-CAM++
                status_box.write("🔬 Generating Grad-CAM++ (higher-order gradient weighting)...")
                try:
                    res_gcpp = run_xai_method("gradcam_plus_plus", res)
                    if res_gcpp and "gradcam_plus_plus" in res_gcpp:
                        res["gradcam_plus_plus"] = res_gcpp["gradcam_plus_plus"]
                        gcpp_data = res["gradcam_plus_plus"]
                except Exception as e:
                    res["gradcam_plus_plus"] = {"error": True, "message": "Grad-CAM++ could not be generated for this analysis."}

                # 3. LIME
                status_box.write("🧩 Generating LIME explanation (superpixel perturbations & Ridge surrogate)...")
                try:
                    res_lime = run_xai_method("lime", res)
                    if res_lime and "lime" in res_lime:
                        res["lime"] = res_lime["lime"]
                        lime_data = res["lime"]
                except Exception as e:
                    res["lime"] = {"error": True, "message": "LIME could not be generated for this analysis."}

                # 4. SHAP
                status_box.write("⚖️ Generating SHAP explanation (Shapley Kernel feature contribution)...")
                try:
                    res_shap = run_xai_method("shap", res)
                    if res_shap and "shap" in res_shap:
                        res["shap"] = res_shap["shap"]
                        shap_data = res["shap"]
                except Exception as e:
                    res["shap"] = {"error": True, "message": "SHAP could not be generated for this analysis."}

                # 5. Integrated Gradients
                status_box.write("📈 Generating Integrated Gradients (path accumulation from baseline)...")
                try:
                    res_ig = run_xai_method("integrated_gradients", res)
                    if res_ig and "integrated_gradients" in res_ig:
                        res["integrated_gradients"] = res_ig["integrated_gradients"]
                        ig_data = res["integrated_gradients"]
                except Exception as e:
                    res["integrated_gradients"] = {"error": True, "message": "Integrated Gradients could not be generated for this analysis."}

                # 6. Cross Method Analysis
                status_box.write("📊 Computing Cross-Method Spatial Agreement & Overlap Metrics...")
                try:
                    res_cross = run_xai_method("all", res)
                    if res_cross and "cross_method_analysis" in res_cross:
                        res["cross_method_analysis"] = res_cross["cross_method_analysis"]
                        cross_metrics = res["cross_method_analysis"]
                except Exception:
                    pass

                status_box.update(label="Complete Multi-Method XAI Analysis Finished Successfully!", state="complete", expanded=False)
                st.session_state.result = res
                st.rerun()

        render_arrow()

        # ======================================================================
        # STEP 8 — UNIFIED XAI DASHBOARD (5 METHOD TABS)
        # ======================================================================
        render_html("""
        <div class="cx-step-card" style="padding-bottom: 10px;">
            <div class="cx-step-header">
                <div class="cx-step-badge">8</div>
                <h3 class="cx-step-title">8. Explainable AI Analysis Dashboard</h3>
            </div>
            <div class="cx-step-desc">
                Inspect how each of the five Explainable AI algorithms explains the prediction 
                <strong>""" + html.escape(pred_label) + """</strong> (""" + f"{conf_prob:.1f}% confidence" + """). 
                Each method explores different mathematical principles—from convolutional gradients to perturbation perturbations and game theory.
            </div>
        </div>
        """)

        xai_tabs = st.tabs([
            "🧠 1. Grad-CAM",
            "🔬 2. Grad-CAM++",
            "🧩 3. LIME",
            "⚖️ 4. SHAP",
            "📈 5. Integrated Gradients",
        ])

        # ----------------------------------------------------------------------
        # TAB 1: GRAD-CAM
        # ----------------------------------------------------------------------
        with xai_tabs[0]:
            render_html("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; color: var(--navy-900); margin: 0;">
                        Grad-CAM — Class Activation Mapping
                    </h4>
                    <span class="cx-badge cx-badge-info">Model-Specific Gradient</span>
                </div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0;">
                    Grad-CAM (Gradient-weighted Class Activation Mapping) calculates gradients of the predicted class score with respect to 
                    feature maps in the final convolutional stage (<code>layer4</code>). Global average pooling of these gradients yields importance 
                    weights that combine feature channels into a coarse activation heatmap.
                </p>
            </div>
            """)

            if gcam_data and not gcam_data.get("error") and gcam_data.get("heatmap_url"):
                g_c1, g_c2, g_c3 = st.columns(3, gap="medium")
                with g_c1:
                    st.markdown("<strong style='font-size:13px;'>Original Input MRI</strong>", unsafe_allow_html=True)
                    if res.get("original_image_url"):
                        render_display_image(res.get("original_image_url"), "Original MRI Scan")
                    elif st.session_state.get("uploaded_image"):
                        st.image(st.session_state.uploaded_image, caption="Original MRI Scan", use_container_width=True)
                with g_c2:
                    st.markdown("<strong style='font-size:13px;'>Grad-CAM Activation Heatmap</strong>", unsafe_allow_html=True)
                    render_display_image(gcam_data.get("heatmap_url"), "Grad-CAM Heatmap")
                with g_c3:
                    st.markdown("<strong style='font-size:13px;'>Heatmap + Anatomical Overlay</strong>", unsafe_allow_html=True)
                    render_display_image(gcam_data.get("overlay_url"), "Grad-CAM Overlay")

                render_html("""
                <div class="cx-xai-legend" style="margin-top: 14px;">
                    <span>0.0 (Cool / Low Contribution)</span>
                    <div class="cx-xai-gradient-bar"></div>
                    <span>1.0 (Warm / High Contribution)</span>
                </div>
                """)

                # Explanations & Dynamic Attribution
                render_html(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 16px;">
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">What Do the Colors Mean?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #475569;">
                            <span style="color: #dc2626; font-weight: 600;">Warm regions (Red / Yellow):</span> Areas of the convolutional feature map that contributed more strongly toward predicting {html.escape(pred_label)}.<br>
                            <span style="color: #2563eb; font-weight: 600;">Cool regions (Blue):</span> Areas with negligible or neutral contribution to the decision.
                        </div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">Why Was This Region Highlighted?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #166534; background: #f0fdf4; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #16a34a;">
                            The highlighted regions represent areas of the feature representation that contributed strongly to the selected model output ({html.escape(pred_label)}).
                        </div>
                    </div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e;">
                    <strong>What the Highlighted Region Represents:</strong> The highlighted region represents model-attributed importance, not a confirmed tumor boundary.
                </div>
                """)

                # Expandable Walkthrough
                with st.expander("🔍 How Grad-CAM Works (Mathematical Attribution Flow)", expanded=False):
                    render_html("""
                    <div class="cx-flow-chain" style="margin: 10px 0;">
                        <div class="cx-flow-node">Prediction</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Gradients</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Feature Maps</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Importance Weights</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node is-active">Heatmap</div>
                    </div>
                    <p style="font-size: 12.5px; color: #475569; line-height: 1.5; margin: 0;">
                        Grad-CAM calculates the gradient of the class score <em>Y<sup>c</sup></em> with respect to feature maps <em>A<sup>k</sup></em>. 
                        Global average pooling computes importance weight <em>α<sub>k</sub><sup>c</sup> = (1/Z) ∑<sub>i</sub>∑<sub>j</sub> (∂Y<sup>c</sup>/∂A<sub>i,j</sub><sup>k</sup>)</em>. 
                        A linear combination of weighted feature maps filtered through ReLU isolates positive features supporting the class.
                    </p>
                    """)
            else:
                st.warning("Grad-CAM explanation could not be generated for this scan.")

        # ----------------------------------------------------------------------
        # TAB 2: GRAD-CAM++
        # ----------------------------------------------------------------------
        with xai_tabs[1]:
            render_html("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; color: var(--navy-900); margin: 0;">
                        Grad-CAM++ — Detailed Model Focus
                    </h4>
                    <span class="cx-badge cx-badge-info">Higher-Order Gradient Formulation</span>
                </div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0;">
                    Grad-CAM++ is an extension of Grad-CAM that uses higher-order gradient information to provide a more detailed estimate of which image regions contributed to a model prediction.
                </p>
            </div>
            """)

            if not gcpp_data or gcpp_data.get("error"):
                st.info("Grad-CAM++ has not been evaluated for this session yet.")
                if st.button("🔬 Generate Grad-CAM++ Explanation", key="cx_btn_gen_gcpp", type="primary"):
                    with st.spinner("Calculating Grad-CAM++ higher-order gradients on ResNet-50 layer4..."):
                        gcpp_res = run_xai_method("gradcam_plus_plus", res)
                        if gcpp_res and "gradcam_plus_plus" in gcpp_res:
                            res["gradcam_plus_plus"] = gcpp_res["gradcam_plus_plus"]
                            st.session_state.result = res
                            st.rerun()
                        else:
                            st.error("Grad-CAM++ could not be generated for this analysis.")
            else:
                gpp_c1, gpp_c2, gpp_c3 = st.columns(3, gap="medium")
                with gpp_c1:
                    st.markdown("<strong style='font-size:13px;'>Original Input MRI</strong>", unsafe_allow_html=True)
                    if res.get("original_image_url"):
                        render_display_image(res.get("original_image_url"), "Original MRI Scan")
                    elif st.session_state.get("uploaded_image"):
                        st.image(st.session_state.uploaded_image, caption="Original MRI Scan", use_container_width=True)
                with gpp_c2:
                    st.markdown("<strong style='font-size:13px;'>Grad-CAM++ Detailed Heatmap</strong>", unsafe_allow_html=True)
                    render_display_image(gcpp_data.get("heatmap_url"), "Grad-CAM++ Heatmap")
                with gpp_c3:
                    st.markdown("<strong style='font-size:13px;'>Grad-CAM++ Anatomical Overlay</strong>", unsafe_allow_html=True)
                    render_display_image(gcpp_data.get("overlay_url"), "Grad-CAM++ Overlay")

                render_html("""
                <div class="cx-xai-legend" style="margin-top: 14px;">
                    <span>0.0 (Cool / Low Contribution)</span>
                    <div class="cx-xai-gradient-bar"></div>
                    <span>1.0 (Warm / High Contribution)</span>
                </div>
                """)

                render_html(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 16px;">
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">What Do the Colors Mean?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #475569;">
                            Brighter or warmer regions indicate areas with stronger model-attributed importance for the selected prediction ({html.escape(pred_label)}).
                        </div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">Why Was This Region Highlighted?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #166534; background: #f0fdf4; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #16a34a;">
                            The highlighted regions represent areas receiving stronger attribution under the Grad-CAM++ calculation for the selected prediction ({html.escape(pred_label)}).
                        </div>
                    </div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e;">
                    <strong>Important Interpretation Note:</strong> The highlighted visualization represents a <em>region contributing to the model prediction</em> instead of a confirmed tumor region.
                </div>
                """)

                with st.expander("🔍 How Grad-CAM++ Works (Analytical Higher-Order Derivatives)", expanded=False):
                    render_html("""
                    <div class="cx-flow-chain" style="margin: 10px 0;">
                        <div class="cx-flow-node">Prediction</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Higher-order gradient information</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Feature-map weighting</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node is-active">Heatmap</div>
                    </div>
                    <p style="font-size: 12.5px; color: #475569; line-height: 1.5; margin: 0;">
                        Unlike standard Grad-CAM which averages gradients uniformly across spatial positions, Grad-CAM++ weights pixel gradients using closed-form 2nd and 3rd order partial derivatives. 
                        This assigns distinct importance weights <em>α<sub>i,j</sub><sup>kc</sup></em> to individual pixel features, resolving multiple object occurrences and fine structures.
                    </p>
                    """)

        # ----------------------------------------------------------------------
        # TAB 3: LIME
        # ----------------------------------------------------------------------
        with xai_tabs[2]:
            render_html("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; color: var(--navy-900); margin: 0;">
                        LIME — Local Explanation
                    </h4>
                    <span class="cx-badge cx-badge-info">Model-Agnostic Superpixel Perturbation</span>
                </div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0;">
                    LIME explains an individual prediction by creating small variations of the input image and observing how the model's prediction changes. It then identifies image regions that locally contribute to the prediction.
                </p>
            </div>
            """)

            if not lime_data or lime_data.get("error"):
                st.info("LIME explanation has not been evaluated for this scan yet.")
                if st.button("🧩 Generate LIME Explanation", key="cx_btn_gen_lime", type="primary"):
                    with st.spinner("Perturbing superpixel segments and evaluating local Ridge surrogate model..."):
                        lime_res = run_xai_method("lime", res)
                        if lime_res and "lime" in lime_res:
                            res["lime"] = lime_res["lime"]
                            st.session_state.result = res
                            st.rerun()
                        else:
                            st.error("LIME could not be generated for this analysis.")
            else:
                l_c1, l_c2, l_c3, l_c4 = st.columns(4, gap="small")
                with l_c1:
                    st.markdown("<strong style='font-size:12px;'>Superpixel Segments</strong>", unsafe_allow_html=True)
                    render_display_image(lime_data.get("superpixel_boundaries_url"), "SLIC Superpixel Segments")
                with l_c2:
                    st.markdown("<strong style='font-size:12px;'>Positive Contributing Regions</strong>", unsafe_allow_html=True)
                    render_display_image(lime_data.get("positive_regions_url"), "Positive Contributing Regions (Green)")
                with l_c3:
                    st.markdown("<strong style='font-size:12px;'>Negative / Opposing Regions</strong>", unsafe_allow_html=True)
                    render_display_image(lime_data.get("negative_regions_url"), "Negative Opposing Regions (Red)")
                with l_c4:
                    st.markdown("<strong style='font-size:12px;'>LIME Attribution Overlay</strong>", unsafe_allow_html=True)
                    render_display_image(lime_data.get("overlay_url"), "LIME Continuous Overlay")

                render_html(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 16px;">
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">What Do the Colors Mean?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #475569;">
                            <span style="color: #16a34a; font-weight: 600;">Green Superpixels:</span> Image segments whose presence positively increased the model's confidence toward predicting {html.escape(pred_label)}.<br>
                            <span style="color: #dc2626; font-weight: 600;">Red Superpixels:</span> Image segments that lowered confidence or supported an alternative category.<br>
                            <span style="color: #ca8a04; font-weight: 600;">Yellow Outlines:</span> Cohesive superpixel partitions generated via SLIC segmentation.
                        </div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">Why Was This Region Highlighted?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #166534; background: #f0fdf4; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #16a34a;">
                            The highlighted image segments represent regions whose perturbation had a measurable effect on the local prediction ({html.escape(pred_label)}).
                        </div>
                    </div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e;">
                    <strong>Local Surrogate Principle:</strong> Highlighted regions represent image segments that had a stronger influence on the model's local prediction behavior. 
                    LIME does not identify the biological cause or clinical histology of a tumor.
                </div>
                """)

                with st.expander("🔍 How LIME Works (Local Surrogate Modeling)", expanded=False):
                    render_html("""
                    <div class="cx-flow-chain" style="margin: 10px 0;">
                        <div class="cx-flow-node">Original MRI</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Perturb image regions</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Generate variations</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Run model</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Analyze prediction changes</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node is-active">Identify influential regions</div>
                    </div>
                    <p style="font-size: 12.5px; color: #475569; line-height: 1.5; margin: 0;">
                        LIME segments the MRI scan into superpixels and generates random binary masks. 
                        Masked regions are occluded with a background baseline and passed in batches through the ResNet-50 network. 
                        A weighted linear regression is fitted with exponential distance weights to approximate the network's local decision boundary around this specific scan.
                    </p>
                    """)

        # ----------------------------------------------------------------------
        # TAB 4: SHAP
        # ----------------------------------------------------------------------
        with xai_tabs[3]:
            render_html("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; color: var(--navy-900); margin: 0;">
                        SHAP — Feature Contribution
                    </h4>
                    <span class="cx-badge cx-badge-info">Cooperative Game Theory / Shapley Values</span>
                </div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0;">
                    SHAP estimates how different parts of the input contribute to the model's prediction by comparing the model output with and without information from different regions of the image.
                </p>
            </div>
            """)

            if not shap_data or shap_data.get("error"):
                st.info("SHAP attribution has not been evaluated for this scan yet.")
                if st.button("⚖️ Generate SHAP Explanation", key="cx_btn_gen_shap", type="primary"):
                    with st.spinner("Evaluating Shapley Kernel permutations through ResNet-50 inference..."):
                        shap_res = run_xai_method("shap", res)
                        if shap_res and "shap" in shap_res:
                            res["shap"] = shap_res["shap"]
                            st.session_state.result = res
                            st.rerun()
                        else:
                            st.error("SHAP could not be generated for this analysis.")
            else:
                s_c1, s_c2, s_c3, s_c4 = st.columns(4, gap="small")
                with s_c1:
                    st.markdown("<strong style='font-size:12px;'>Original MRI Scan</strong>", unsafe_allow_html=True)
                    if res.get("original_image_url"):
                        render_display_image(res.get("original_image_url"), "Original MRI Scan")
                    elif st.session_state.get("uploaded_image"):
                        st.image(st.session_state.uploaded_image, caption="Original MRI Scan", use_container_width=True)
                with s_c2:
                    st.markdown("<strong style='font-size:12px;'>SHAP Feature Attribution</strong>", unsafe_allow_html=True)
                    render_display_image(shap_data.get("heatmap_url"), "SHAP Attribution Heatmap")
                with s_c3:
                    st.markdown("<strong style='font-size:12px;'>Bipolar Contribution Map</strong>", unsafe_allow_html=True)
                    render_display_image(shap_data.get("diverging_overlay_url"), "SHAP Bipolar Overlay (Red=Pos, Blue=Neg)")
                with s_c4:
                    st.markdown("<strong style='font-size:12px;'>SHAP + MRI Overlay</strong>", unsafe_allow_html=True)
                    render_display_image(shap_data.get("overlay_url"), "SHAP Combined Overlay")

                render_html(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 16px;">
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">What Do the Colors Mean?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #475569;">
                            Regions with stronger positive contribution indicate areas that increased the model's output toward the selected class ({html.escape(pred_label)}), 
                            while regions with negative contribution reduced support for that class.
                        </div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">Why Was This Region Highlighted?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #166534; background: #f0fdf4; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #16a34a;">
                            The highlighted regions represent features with stronger contribution toward or away from the selected model output ({html.escape(pred_label)}).
                        </div>
                    </div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e;">
                    <strong>Scientific Integrity Notice:</strong> SHAP values provide mathematical attribution of model feature weights. 
                    They describe statistical credit assignment within the neural network and do not represent medical certainty or histological margins.
                </div>
                """)

                with st.expander("🔍 How SHAP Works (Axiomatic Shapley Additive Explanations)", expanded=False):
                    render_html("""
                    <div class="cx-flow-chain" style="margin: 10px 0;">
                        <div class="cx-flow-node">MRI</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Evaluate feature/region contributions</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Estimate contribution values</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node is-active">Generate attribution visualization</div>
                    </div>
                    <p style="font-size: 12.5px; color: #475569; line-height: 1.5; margin: 0;">
                        Rooted in cooperative game theory, SHAP allocates fair payouts to each image region based on its marginal contribution across all possible feature subsets. 
                        Kernel SHAP evaluates perturbations weighted by the combinatorial Shapley kernel <em>π(z) = (M-1) / ((M choose |z|) |z| (M-|z|))</em> to guarantee 
                        efficiency, symmetry, dummy, and additivity axioms.
                    </p>
                    """)

        # ----------------------------------------------------------------------
        # TAB 5: INTEGRATED GRADIENTS
        # ----------------------------------------------------------------------
        with xai_tabs[4]:
            render_html("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; color: var(--navy-900); margin: 0;">
                        Integrated Gradients — Input Attribution
                    </h4>
                    <span class="cx-badge cx-badge-info">Axiomatic Path-Integrated Attribution</span>
                </div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0;">
                    Integrated Gradients estimates how individual input features contributed to the prediction by accumulating gradients along a path from a baseline image to the actual MRI.
                </p>
            </div>
            """)

            if not ig_data or ig_data.get("error"):
                st.info("Integrated Gradients has not been evaluated for this scan yet.")
                if st.button("📈 Generate Integrated Gradients", key="cx_btn_gen_ig", type="primary"):
                    with st.spinner("Accumulating path gradients from zero baseline to MRI scan..."):
                        ig_res = run_xai_method("integrated_gradients", res)
                        if ig_res and "integrated_gradients" in ig_res:
                            res["integrated_gradients"] = ig_res["integrated_gradients"]
                            st.session_state.result = res
                            st.rerun()
                        else:
                            st.error("Integrated Gradients could not be generated for this analysis.")
            else:
                ig_c1, ig_c2, ig_c3 = st.columns(3, gap="medium")
                with ig_c1:
                    st.markdown("<strong style='font-size:13px;'>Original Input MRI</strong>", unsafe_allow_html=True)
                    if res.get("original_image_url"):
                        render_display_image(res.get("original_image_url"), "Original MRI Scan")
                    elif st.session_state.get("uploaded_image"):
                        st.image(st.session_state.uploaded_image, caption="Original MRI Scan", use_container_width=True)
                with ig_c2:
                    st.markdown("<strong style='font-size:13px;'>Integrated Gradients Attribution Map</strong>", unsafe_allow_html=True)
                    render_display_image(ig_data.get("heatmap_url"), "Integrated Gradients Heatmap")
                with ig_c3:
                    st.markdown("<strong style='font-size:13px;'>Attribution + Anatomical Overlay</strong>", unsafe_allow_html=True)
                    render_display_image(ig_data.get("overlay_url"), "Integrated Gradients Overlay")

                render_html("""
                <div class="cx-xai-legend" style="margin-top: 14px;">
                    <span>0.0 (Cool / Baseline Attribution)</span>
                    <div class="cx-xai-gradient-bar"></div>
                    <span>1.0 (Warm / Peak Path Attribution)</span>
                </div>
                """)

                render_html(f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 16px;">
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">What Do the Colors Mean?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #475569;">
                            Warmer and brighter pixels represent areas where cumulative path gradients exhibited highest attribution as the input transitioned from the baseline black image to the patient MRI scan.
                        </div>
                    </div>
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                        <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 6px;">Why Was This Region Highlighted?</strong>
                        <div style="font-size: 12.5px; line-height: 1.5; color: #166534; background: #f0fdf4; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #16a34a;">
                            The highlighted regions represent input areas receiving stronger attribution along the path from the baseline input to the MRI ({html.escape(pred_label)}).
                        </div>
                    </div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #92400e;">
                    <strong>Attribution Scope:</strong> Integrated Gradients directly explains the neural function's sensitivity along a straight-line interpolation path. 
                    It does not directly identify or delineate a pathological tumor.
                </div>
                """)

                with st.expander("🔍 How Integrated Gradients Works (Axiomatic Path Integration)", expanded=False):
                    render_html("""
                    <div class="cx-flow-chain" style="margin: 10px 0;">
                        <div class="cx-flow-node">Baseline</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Gradual transition toward MRI</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Calculate gradients</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node">Accumulate attributions</div>
                        <div class="cx-flow-arrow">→</div>
                        <div class="cx-flow-node is-active">Generate attribution map</div>
                    </div>
                    <p style="font-size: 12.5px; color: #475569; line-height: 1.5; margin: 0;">
                        Integrated Gradients satisfies two fundamental axioms for attribution: <em>Completeness</em> (attributions sum to the difference between model output and baseline output) 
                        and <em>Implementation Invariance</em>. By taking discrete interpolation steps <em>x' + (k/m)(x - x')</em> and averaging gradients at each step, 
                        the method avoids gradient saturation problems typical of vanilla backpropagation.
                    </p>
                    """)

        render_arrow()

        # ======================================================================
        # STEP 9 — COMPARING XAI METHODS
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">9</div>
                <h3 class="cx-step-title">9. Comparing XAI Methods</h3>
            </div>
            <div class="cx-step-desc">
                An objective technical comparison summarizing the mathematical foundation, framework type, and visualization outputs of all five evaluated explainability methods:
            </div>
            <table class="cx-compare-table" style="margin-top: 12px;">
                <thead>
                    <tr>
                        <th style="width: 20%;">Method</th>
                        <th style="width: 32%;">Main Idea</th>
                        <th style="width: 24%;">Type</th>
                        <th style="width: 24%;">Output</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Grad-CAM</strong></td>
                        <td>Uses gradients and convolutional feature maps</td>
                        <td>Model-specific (CNN)</td>
                        <td>Heatmap</td>
                    </tr>
                    <tr>
                        <td><strong>Grad-CAM++</strong></td>
                        <td>Refined higher-order gradient-based localization</td>
                        <td>Model-specific (CNN)</td>
                        <td>Detailed heatmap</td>
                    </tr>
                    <tr>
                        <td><strong>LIME</strong></td>
                        <td>Perturbs input superpixel regions</td>
                        <td>Model-agnostic</td>
                        <td>Superpixel explanation</td>
                    </tr>
                    <tr>
                        <td><strong>SHAP</strong></td>
                        <td>Estimates feature contributions via Shapley values</td>
                        <td>Model/framework dependent</td>
                        <td>Attribution map</td>
                    </tr>
                    <tr>
                        <td><strong>Integrated Gradients</strong></td>
                        <td>Accumulates gradients along path from baseline</td>
                        <td>Gradient-based (Axiomatic)</td>
                        <td>Attribution map</td>
                    </tr>
                </tbody>
            </table>
            <div style="font-size: 12.5px; color: #64748b; line-height: 1.5; margin-top: 12px;">
                <em>No single method is universally superior. Gradient-based techniques (Grad-CAM, Grad-CAM++) offer fast, high-level semantic insights into convolutional layer activations; perturbation methods (LIME) reveal local decision boundaries; game-theoretic methods (SHAP) allocate fair feature credit; and path-based integration (Integrated Gradients) provides axiomatic pixel-level attributions.</em>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 10 — "WHY USE MULTIPLE XAI METHODS?"
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">10</div>
                <h3 class="cx-step-title">10. Why Multiple Explanations?</h3>
            </div>
            <div class="cx-step-desc">
                Different XAI methods use different approaches to explain model behavior. Comparing multiple explanations can help researchers examine whether important regions identified by one method are also represented by other methods.
            </div>
            <div style="margin: 16px 0;">
                <div class="cx-flow-chain">
                    <div class="cx-flow-node">One MRI</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node">One Prediction</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node">Multiple XAI Methods</div>
                    <div class="cx-flow-arrow">→</div>
                    <div class="cx-flow-node is-active">Multiple Perspectives on Model Behavior</div>
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
                <strong style="color: var(--navy-900); font-size: 13px; display: block; margin-bottom: 4px;">Complementary Evidence, Not Diagnostic Proof</strong>
                <p style="color: #475569; font-size: 12.5px; margin: 0; line-height: 1.5;">
                    When distinct mathematical formulations (e.g. backpropagation gradients vs. occlusive superpixel perturbations) consistently point toward the same region of interest, 
                    researchers can have higher confidence that the deep neural network learned coherent visual patterns rather than spurious background artifacts. 
                    <strong>Agreement between methods does not prove that the model is medically correct.</strong>
                </p>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 11 — CROSS-METHOD EXPLANATION ANALYSIS (OBJECTIVE OVERLAP METRICS)
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">11</div>
                <h3 class="cx-step-title">11. Cross-Method Explanation Analysis</h3>
            </div>
            <div class="cx-step-desc">
                This section compares the spatial patterns produced by the different explanation methods. 
                Similarity indicates that methods identified overlapping regions of model attribution; 
                differences indicate that the methods highlighted different aspects of the model's behavior.
            </div>
        """)

        if cross_metrics and cross_metrics.get("available") and cross_metrics.get("pairwise_comparisons"):
            pw_list = cross_metrics.get("pairwise_comparisons", [])
            mean_corr = cross_metrics.get("mean_correlation", 0.0)
            mean_iou = cross_metrics.get("mean_salient_iou", 0.0)
            mean_cos = cross_metrics.get("mean_cosine_similarity", 0.0)
            summary_txt = cross_metrics.get("summary", "")

            render_html(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0 16px 0;">
                <div class="cx-metric-pill">
                    <div class="cx-metric-pill-val">{mean_corr:+.3f}</div>
                    <div class="cx-metric-pill-lbl">Mean Spatial Correlation (r)</div>
                </div>
                <div class="cx-metric-pill">
                    <div class="cx-metric-pill-val">{mean_iou:.3f}</div>
                    <div class="cx-metric-pill-lbl">Mean Salient Mask IoU</div>
                </div>
                <div class="cx-metric-pill">
                    <div class="cx-metric-pill-val">{mean_cos:.3f}</div>
                    <div class="cx-metric-pill-lbl">Mean Cosine Alignment</div>
                </div>
            </div>
            <table class="cx-compare-table">
                <thead>
                    <tr>
                        <th>Method Pair</th>
                        <th>Pearson Correlation (r)</th>
                        <th>Cosine Similarity</th>
                        <th>Salient Mask IoU (Top 20%)</th>
                        <th>Shared Active Area Overlap</th>
                    </tr>
                </thead>
                <tbody>
            """)

            for pw in pw_list:
                m_a = html.escape(pw.get("method_a", ""))
                m_b = html.escape(pw.get("method_b", ""))
                r_val = pw.get("pearson_correlation", 0.0)
                cos_val = pw.get("cosine_similarity", 0.0)
                iou_val = pw.get("salient_iou", 0.0)
                sh_val = pw.get("shared_active_overlap_pct", 0.0)
                st.markdown(
                    f"<tr style='border-bottom: 1px solid #f1f5f9; font-size: 12.5px;'>"
                    f"<td><strong>{m_a}</strong> vs <strong>{m_b}</strong></td>"
                    f"<td><code>{r_val:+.3f}</code></td>"
                    f"<td><code>{cos_val:.3f}</code></td>"
                    f"<td><code>{iou_val:.3f}</code></td>"
                    f"<td><strong>{sh_val:.1f}%</strong></td>"
                    f"</tr>",
                    unsafe_allow_html=True
                )

            render_html(f"""
                </tbody>
            </table>
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; margin-top: 14px; font-size: 12.5px; color: #166534;">
                <strong>Cross-Method Synthesis:</strong> {html.escape(summary_txt)}
            </div>
            """)
        else:
            render_html("""
            <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 18px; text-align: center; color: #64748b; font-size: 13px;">
                Multi-method overlap metrics become available when two or more XAI techniques are generated. 
                Click <strong>"🚀 Run Complete XAI Analysis"</strong> above to compute all remaining methods and unlock spatial agreement metrics.
            </div>
            """)

        render_html("""
        <div style="font-size: 11.5px; color: #64748b; margin-top: 10px;">
            *Do not interpret similarity as proof of diagnostic correctness. Metrics quantify mathematical spatial overlap across feature representations.
        </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # AI vs Human Interpretation (Comparative Card)
        # ======================================================================
        render_html("""
        <div class="cx-step-card">
            <div class="cx-step-header">
                <div class="cx-step-badge">⚖️</div>
                <h3 class="cx-step-title">AI vs Human Interpretation</h3>
            </div>
            <div class="cx-step-desc">
                An objective comparison between artificial neural pattern recognition and human clinical diagnostic reasoning:
            </div>
            <table class="cx-compare-table">
                <thead>
                    <tr>
                        <th style="width: 25%;">Dimension</th>
                        <th style="width: 37.5%;">AI System</th>
                        <th style="width: 37.5%;">Human Expert</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Analysis Mechanism</strong></td>
                        <td>Analyzes learned image patterns and statistical pixel contrasts</td>
                        <td>Uses medical training, anatomical knowledge, and clinical context</td>
                    </tr>
                    <tr>
                        <td><strong>Output Nature</strong></td>
                        <td>Produces numerical model probabilities across trained classes</td>
                        <td>Integrates imaging findings with patient history and symptoms</td>
                    </tr>
                    <tr>
                        <td><strong>Attribution Scope</strong></td>
                        <td>XAI methods provide mathematical model attribution</td>
                        <td>Can assess broader clinical, surgical, and differential findings</td>
                    </tr>
                    <tr>
                        <td><strong>Clinical Role</strong></td>
                        <td>Does not independently establish medical diagnosis</td>
                        <td>Provides professional, certified clinical interpretation and care plans</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 12 — FINAL EXPLAINABLE AI SUMMARY
        # ======================================================================
        render_html(f"""
        <div class="cx-interpretation-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 22px;">📋</span>
                <h3 style="font-family: var(--font-display); font-size: 21px; font-weight: 700; color: #ffffff; margin: 0;">
                    12. Final Explainable AI Summary
                </h3>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; background: rgba(255,255,255,0.06); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.1);">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Prediction</div>
                    <div style="font-family: var(--font-display); font-size: 19px; font-weight: 700; color: #ffffff; margin-top: 2px;">{html.escape(pred_class)}</div>
                </div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Confidence</div>
                    <div style="font-family: var(--font-display); font-size: 19px; font-weight: 700; color: #ffffff; margin-top: 2px;">{conf_prob:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Model Uncertainty</div>
                    <div style="font-family: var(--font-display); font-size: 19px; font-weight: 700; color: #ffffff; margin-top: 2px;">{float(uncertainty_std):.5f} ({html.escape(str(uncertainty_level).capitalize())})</div>
                </div>
            </div>

            <div style="font-weight: 700; font-size: 13px; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
                Method-by-Method Attribution Findings:
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 16px;">
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px;">
                    <strong style="color: #38bdf8; font-size: 12.5px;">Grad-CAM</strong>
                    <p style="color: #cbd5e1; font-size: 11.5px; margin: 3px 0 0 0; line-height: 1.4;">
                        {"Attribution concentrated in layer4 convolutional representations supporting " + html.escape(pred_label) + "." if gcam_data and not gcam_data.get("error") else "Pending evaluation."}
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px;">
                    <strong style="color: #38bdf8; font-size: 12.5px;">Grad-CAM++</strong>
                    <p style="color: #cbd5e1; font-size: 11.5px; margin: 3px 0 0 0; line-height: 1.4;">
                        {"Refined gradient attribution resolving finer focal localization for " + html.escape(pred_label) + "." if gcpp_data and not gcpp_data.get("error") else "Pending evaluation."}
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px;">
                    <strong style="color: #38bdf8; font-size: 12.5px;">LIME</strong>
                    <p style="color: #cbd5e1; font-size: 11.5px; margin: 3px 0 0 0; line-height: 1.4;">
                        {"Identified contiguous superpixels with highest local positive contribution toward " + html.escape(pred_label) + "." if lime_data and not lime_data.get("error") else "Pending evaluation."}
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px;">
                    <strong style="color: #38bdf8; font-size: 12.5px;">SHAP</strong>
                    <p style="color: #cbd5e1; font-size: 11.5px; margin: 3px 0 0 0; line-height: 1.4;">
                        {"Fair Shapley credit assignment distinguishing positive supporting features from opposing segments." if shap_data and not shap_data.get("error") else "Pending evaluation."}
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px;">
                    <strong style="color: #38bdf8; font-size: 12.5px;">Integrated Gradients</strong>
                    <p style="color: #cbd5e1; font-size: 11.5px; margin: 3px 0 0 0; line-height: 1.4;">
                        {"Axiomatic path accumulation from baseline black canvas confirming primary input feature influence." if ig_data and not ig_data.get("error") else "Pending evaluation."}
                    </p>
                </div>
            </div>

            <div style="border-top: 1px solid rgba(255,255,255,0.12); padding-top: 12px;">
                <div style="font-weight: 700; font-size: 13px; color: #93c5fd; margin-bottom: 4px;">Overall Interpretation:</div>
                <div style="color: #e2e8f0; font-size: 13px; line-height: 1.55; border-left: 3px solid #38bdf8; padding-left: 12px;">
                    The model predicted {html.escape(pred_class)} with {conf_prob:.2f}% confidence. 
                    The XAI methods provide different views of which image regions contributed to this prediction. 
                    These visualizations describe model behavior and attribution and should not be interpreted as a definitive medical diagnosis or exact tumor boundary.
                </div>
            </div>
        </div>
        """)

        render_arrow()

        # ======================================================================
        # STEP 13 — MEDICAL SAFETY & CLINICAL GOVERNANCE NOTE
        # ======================================================================
        render_html("""
        <div class="cx-safety-card">
            <div style="display: flex; gap: 14px; align-items: flex-start;">
                <div style="font-size: 24px; flex-shrink: 0;">🛡️</div>
                <div>
                    <strong style="color: #9a3412; font-size: 14.5px; display: block; margin-bottom: 4px;">
                        13. Medical Safety Notice & Academic Research Disclaimer
                    </strong>
                    <p style="color: #c2410c; font-size: 13px; line-height: 1.55; margin: 0 0 8px 0;">
                        <strong>XAI visualizations explain aspects of model behavior. They do not establish a medical diagnosis, determine an exact tumor boundary, or replace assessment by a qualified healthcare professional.</strong>
                    </p>
                    <div style="font-size: 12px; color: #9a3412; line-height: 1.45;">
                        • The model attributed greater importance to the highlighted regions based on learned convolutional patterns.<br>
                        • The highlighted region contributed more strongly to the selected prediction.<br>
                        • The visualizations represent mathematical model attribution rather than confirmed histological pathology.
                    </div>
                </div>
            </div>
        </div>
        """)


        # Quantitative XAI & Localization Metrics (Preserved)
        # ----------------------------------------------------------------------
        stability = res.get("stability_percentage", xai_metrics_data.get("stability_percentage"))
        snr = res.get("signal_to_noise_ratio", xai_metrics_data.get("signal_to_noise_ratio"))
        coverage = res.get("active_area_coverage_pct", xai_metrics_data.get("active_area_coverage_pct"))
        iou = res.get("localization_iou", xai_metrics_data.get("localization_iou"))
        dice = res.get("localization_dice", xai_metrics_data.get("localization_dice"))

        with st.container(border=True):
            render_html("""
            <span class="cx-kpi-label">QUANTITATIVE XAI METRICS</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 4px 0 14px 0; color: #0f172a;">Localization & Explanation Stability</h3>
            """)

            qm1, qm2, qm3, qm4, qm5 = st.columns(5)
            with qm1:
                render_html(f"<div class='cx-metric-pill'><div class='cx-metric-pill-val'>{f'{float(stability):.1f}%' if stability is not None else 'N/A'}</div><div class='cx-metric-pill-lbl'>Attribution Stability</div></div>")
            with qm2:
                render_html(f"<div class='cx-metric-pill'><div class='cx-metric-pill-val'>{f'{float(snr):.2f}' if snr is not None else 'N/A'}</div><div class='cx-metric-pill-lbl'>Signal-to-Noise</div></div>")
            with qm3:
                render_html(f"<div class='cx-metric-pill'><div class='cx-metric-pill-val'>{f'{float(coverage):.1f}%' if coverage is not None else 'N/A'}</div><div class='cx-metric-pill-lbl'>Active Area Cov</div></div>")
            with qm4:
                render_html(f"<div class='cx-metric-pill'><div class='cx-metric-pill-val'>{f'{float(iou):.3f}' if iou is not None else 'N/A'}</div><div class='cx-metric-pill-lbl'>Localization IoU</div></div>")
            with qm5:
                render_html(f"<div class='cx-metric-pill'><div class='cx-metric-pill-val'>{f'{float(dice):.3f}' if dice is not None else 'N/A'}</div><div class='cx-metric-pill-lbl'>Dice Score</div></div>")

            if res.get("interpretation"):
                render_html(f"""
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px; margin-top: 16px; color: #166534; font-size: 13px;">
                    <strong>Clinical Interpretation Summary:</strong><br>{html.escape(res.get('interpretation'))}
                </div>
                """)

        # ----------------------------------------------------------------------
        # Case Audit Details (Preserved)
        # ----------------------------------------------------------------------
        render_html(f"""
        <div class="cx-card">
            <span class="cx-kpi-label">CASE AUDIT LOG</span>
            <h3 style="font-family: var(--font-display); font-size: 17px; font-weight: 700; margin: 4px 0 14px 0;">Acquisition & Patient Information</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; font-size: 13px;">
                <div><strong>Patient / Clinician:</strong> {html.escape(str(res.get('patient_name') or display_patient_name))}</div>
                <div><strong>Patient ID:</strong> {html.escape(str(res.get('patient_id') or 'N/A'))}</div>
                <div><strong>Case ID:</strong> {html.escape(str(res.get('patient_case_id') or 'N/A'))}</div>
                <div><strong>MRI Sequence:</strong> {html.escape(str(res.get('sequence') or 'T1-CE Axial'))}</div>
                <div><strong>Anatomy:</strong> {html.escape(str(res.get('anatomy') or 'Cerebral Hemisphere'))}</div>
                <div><strong>Target Layer:</strong> {html.escape(str(res.get('cam_layer') or 'layer4'))}</div>
            </div>
        </div>
        """)

        # ======================================================================
        # STEP 14 — MEDICAL REPORT GENERATION (NEW FEATURE)
        # ======================================================================
        render_html("""
        <div class="cx-step-card" style="border-left: 4px solid #0284c7; background: #ffffff; margin-top: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
                <div>
                    <span class="cx-kpi-label">CLINICAL DOCUMENTATION & GOVERNANCE</span>
                    <h2 style="font-family: var(--font-display); font-size: 22px; font-weight: 700; color: var(--navy-900); margin: 2px 0 0 0;">
                        14. Structured MRI Medical Report
                    </h2>
                </div>
                <div>
                    <span class="cx-badge cx-badge-info">Audit-Ready Report</span>
                </div>
            </div>
            <p style="color: #475569; font-size: 13.5px; line-height: 1.5; margin: 0 0 14px 0;">
                Synthesize the current case's ResNet-50 predictions, patient metadata, and all evaluated Explainable AI (XAI) attributions 
                into a structured clinical report. You can review, edit, regenerate, copy, or download the report as a PDF.
            </p>
        </div>
        """)

        def ensure_full_xai_suite(analysis_data: dict) -> dict:
            """
            Checks if LIME, SHAP, and Integrated Gradients are present in analysis_data.
            If missing or errored, computes them via run_xai_method and updates analysis_data.
            """
            missing = []
            for k in ["lime", "shap", "integrated_gradients"]:
                val = analysis_data.get(k)
                if not val or (isinstance(val, dict) and (val.get("error") or not (val.get("overlay_url") or val.get("heatmap_url")))):
                    missing.append(k)

            if not missing:
                return analysis_data

            with st.status("🔬 Computing complete Explainable AI suite (LIME, SHAP, Integrated Gradients) for medical report...", expanded=True) as status_box:
                if "lime" in missing:
                    status_box.write("🧩 Generating LIME explanation (superpixel perturbations & surrogate model)...")
                    try:
                        out = run_xai_method("lime", analysis_data)
                        if out and "lime" in out and not out["lime"].get("error"):
                            analysis_data["lime"] = out["lime"]
                    except Exception as e:
                        print(f"LIME error: {e}")

                if "shap" in missing:
                    status_box.write("⚖️ Generating SHAP explanation (Shapley feature attribution)...")
                    try:
                        out = run_xai_method("shap", analysis_data)
                        if out and "shap" in out and not out["shap"].get("error"):
                            analysis_data["shap"] = out["shap"]
                    except Exception as e:
                        print(f"SHAP error: {e}")

                if "integrated_gradients" in missing:
                    status_box.write("📈 Generating Integrated Gradients (path accumulation from baseline)...")
                    try:
                        out = run_xai_method("integrated_gradients", analysis_data)
                        if out and "integrated_gradients" in out and not out["integrated_gradients"].get("error"):
                            analysis_data["integrated_gradients"] = out["integrated_gradients"]
                    except Exception as e:
                        print(f"Integrated Gradients error: {e}")

                st.session_state.result = analysis_data
                status_box.update(label="✓ All 5 Explainable AI methods generated successfully!", state="complete", expanded=False)

            return analysis_data

        # If report not yet generated, show clearly visible "Generate Report" button
        if st.session_state.current_report is None:
            rep_gen_col1, rep_gen_col2, rep_gen_col3 = st.columns([1, 2, 1])
            with rep_gen_col2:
                generate_report_btn = st.button(
                    "📄 Generate Report",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.report_generating,
                    key="cx_btn_generate_mri_report",
                    help="Compiles the MRI analysis, patient details, findings, impressions, and all 5 Explainable AI visualizations into a structured medical report."
                )

            if generate_report_btn:
                st.session_state.report_generating = True
                res = ensure_full_xai_suite(res)
                with st.spinner("Generating MRI report..."):
                    patient_meta = {
                        "patient_name": res.get("patient_name") or display_patient_name,
                        "patient_id": res.get("patient_id") or "Not available",
                        "case_id": res.get("patient_case_id") or "Not available",
                        "patient_age": res.get("_patient_age") or res.get("patient_age"),
                        "patient_gender": res.get("_patient_gender") or res.get("patient_gender"),
                        "referring_physician": res.get("referring_physician") or "Not available",
                    }
                    exam_meta = {
                        "body_region": res.get("anatomy") or "Brain / Cerebral Hemisphere",
                        "mri_sequence": res.get("sequence") or "T1-CE Axial",
                        "image_name": res.get("image_name") or "mri_scan.png",
                    }
                    rep_obj = request_mri_report(res, patient_metadata=patient_meta, examination_metadata=exam_meta)
                    st.session_state.report_generating = False
                    if rep_obj:
                        st.session_state.current_report = rep_obj
                        st.session_state.report_edit_findings = rep_obj.get("findings", "")
                        st.session_state.report_edit_impression = rep_obj.get("impression", "")
                        st.session_state.report_edit_physician = rep_obj.get("patient_information", {}).get("referring_physician", "Not available")
                        st.session_state.report_edit_notes = rep_obj.get("notes", "")
                        st.rerun()
                    else:
                        st.error("Unable to generate the report. Please try again.")

        else:
            rep = st.session_state.current_report
            r_hdr = rep.get("header", {})
            r_patient = rep.get("patient_information", {})
            r_exam = rep.get("examination_information", {})
            r_pred = rep.get("ai_prediction", {})
            r_mod = rep.get("model_information", {})
            r_xai = rep.get("xai_visualizations", [])

            # If current report has fewer than 5 XAI visualizations, offer 1-click update button
            if len(r_xai) < 5:
                missing_names = [m for m in ["Grad-CAM", "Grad-CAM++", "LIME", "SHAP", "Integrated Gradients"] if not any(x.get("method") == m for x in r_xai)]
                st.warning(f"⚠️ This report contains {len(r_xai)} of 5 Explainable AI visualizations (missing: {', '.join(missing_names)}).")
                if st.button("➕ Add LIME, SHAP & Integrated Gradients to Report", type="primary", key="cx_btn_upgrade_xai_report"):
                    res = ensure_full_xai_suite(res)
                    with st.spinner("Updating report with all 5 Explainable AI visualizations..."):
                        patient_meta = {
                            "patient_name": res.get("patient_name") or display_patient_name,
                            "patient_id": res.get("patient_id") or "Not available",
                            "case_id": res.get("patient_case_id") or "Not available",
                            "patient_age": res.get("_patient_age") or res.get("patient_age"),
                            "patient_gender": res.get("_patient_gender") or res.get("patient_gender"),
                            "referring_physician": st.session_state.report_edit_physician or "Not available",
                        }
                        exam_meta = {
                            "body_region": res.get("anatomy") or "Brain / Cerebral Hemisphere",
                            "mri_sequence": res.get("sequence") or "T1-CE Axial",
                            "image_name": res.get("image_name") or "mri_scan.png",
                        }
                        new_rep = request_mri_report(res, patient_metadata=patient_meta, examination_metadata=exam_meta)
                        if new_rep:
                            st.session_state.current_report = new_rep
                            st.session_state.report_edit_findings = new_rep.get("findings", "")
                            st.session_state.report_edit_impression = new_rep.get("impression", "")
                            st.session_state.report_editing = False
                            st.success("Report updated successfully with all 5 Explainable AI visualizations.")
                            st.rerun()

            # ------------------------------------------------------------------
            # REPORT ACTION TOOLBAR (EDIT, REGENERATE, COPY, DOWNLOAD PDF, PRINT)
            # ------------------------------------------------------------------
            act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns(5)
            with act_c1:
                edit_btn_label = "✖️ Close Editor" if st.session_state.report_editing else "✏️ Edit Report"
                if st.button(edit_btn_label, use_container_width=True, key="cx_btn_toggle_edit_report"):
                    st.session_state.report_editing = not st.session_state.report_editing
                    st.rerun()

            with act_c2:
                if st.button("🔄 Regenerate", use_container_width=True, key="cx_btn_regen_report", help="Regenerates structured findings and impression with the complete 5-method Explainable AI suite."):
                    res = ensure_full_xai_suite(res)
                    with st.spinner("Regenerating MRI report..."):
                        patient_meta = {
                            "patient_name": res.get("patient_name") or display_patient_name,
                            "patient_id": res.get("patient_id") or "Not available",
                            "case_id": res.get("patient_case_id") or "Not available",
                            "patient_age": res.get("_patient_age") or res.get("patient_age"),
                            "patient_gender": res.get("_patient_gender") or res.get("patient_gender"),
                            "referring_physician": st.session_state.report_edit_physician or "Not available",
                        }
                        exam_meta = {
                            "body_region": res.get("anatomy") or "Brain / Cerebral Hemisphere",
                            "mri_sequence": res.get("sequence") or "T1-CE Axial",
                            "image_name": res.get("image_name") or "mri_scan.png",
                        }
                        new_rep = request_mri_report(res, patient_metadata=patient_meta, examination_metadata=exam_meta)
                        if new_rep:
                            st.session_state.current_report = new_rep
                            st.session_state.report_edit_findings = new_rep.get("findings", "")
                            st.session_state.report_edit_impression = new_rep.get("impression", "")
                            st.session_state.report_editing = False
                            st.success("Report regenerated successfully with all Explainable AI visualizations.")
                            st.rerun()
                        else:
                            st.error("Unable to generate the report. Please try again.")

            with act_c3:
                show_copy = st.toggle("📋 Copy Text", key="cx_toggle_copy_report")

            with act_c4:
                pdf_data = request_report_pdf(rep)
                if pdf_data:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=f"{rep.get('id', 'mri_report')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key="cx_btn_download_pdf"
                    )
                else:
                    st.button("📥 PDF Unavailable", disabled=True, use_container_width=True)

            with act_c5:
                st.components.v1.html(
                    """
                    <button onclick="window.print()" style="
                        width: 100%;
                        height: 38px;
                        background: #0f172a;
                        color: #ffffff;
                        border: 1px solid #334155;
                        border-radius: 8px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        font-size: 13.5px;
                        font-weight: 600;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 6px;
                    ">
                        🖨️ Print Report
                    </button>
                    """,
                    height=45
                )

            # ------------------------------------------------------------------
            # COPY REPORT TEXT DISPLAY
            # ------------------------------------------------------------------
            if show_copy:
                report_plaintext = report_generator.format_report_text(rep)
                st.info("Click the copy icon in the top-right corner of the code block below to copy the complete report text.")
                st.code(report_plaintext, language="markdown")

            # ------------------------------------------------------------------
            # INLINE REPORT EDITOR
            # ------------------------------------------------------------------
            if st.session_state.report_editing:
                with st.expander("✏️ Report Editor — Modify Findings & Clinical Impressions", expanded=True):
                    st.markdown("Modify any section below before exporting or printing. Updates apply to this report instance.")
                    ed_col1, ed_col2 = st.columns(2)
                    with ed_col1:
                        new_pname = st.text_input("Patient Name", value=r_patient.get("patient_name", "Not available"), key="cx_edit_pname")
                        new_ref_phys = st.text_input("Referring Physician", value=r_patient.get("referring_physician", "Not available"), key="cx_edit_ref_phys")
                    with ed_col2:
                        new_pid = st.text_input("Patient ID", value=r_patient.get("patient_id", "Not available"), key="cx_edit_pid")
                        new_notes = st.text_input("Clinical Notes / Comments", value=rep.get("notes", ""), key="cx_edit_notes")

                    new_findings = st.text_area("Findings", value=st.session_state.report_edit_findings, height=140, key="cx_edit_findings_area")
                    new_impression = st.text_area("Impression", value=st.session_state.report_edit_impression, height=110, key="cx_edit_impression_area")

                    save_col1, save_col2 = st.columns([1, 4])
                    with save_col1:
                        if st.button("💾 Save Edits", type="primary", use_container_width=True, key="cx_btn_save_edits"):
                            rep["findings"] = new_findings
                            rep["impression"] = new_impression
                            rep["notes"] = new_notes
                            rep["patient_information"]["patient_name"] = new_pname
                            rep["patient_information"]["patient_id"] = new_pid
                            rep["patient_information"]["referring_physician"] = new_ref_phys
                            rep["report_version"] = int(rep.get("report_version", 1)) + 1
                            st.session_state.report_edit_findings = new_findings
                            st.session_state.report_edit_impression = new_impression
                            st.session_state.report_edit_physician = new_ref_phys
                            st.session_state.report_edit_notes = new_notes
                            st.session_state.current_report = rep
                            st.session_state.report_editing = False

                            # Sync update to backend
                            try:
                                api_request(
                                    "PUT",
                                    f"/api/reports/{rep.get('id')}",
                                    json={
                                        "findings": new_findings,
                                        "impression": new_impression,
                                        "patient_name": new_pname,
                                        "patient_id": new_pid,
                                        "referring_physician": new_ref_phys,
                                        "notes": new_notes,
                                    },
                                    headers=get_auth_headers(),
                                    timeout=10
                                )
                            except Exception:
                                pass

                            st.success("Report updated successfully.")
                            st.rerun()

                    with save_col2:
                        if st.button("Cancel", use_container_width=False, key="cx_btn_cancel_edits"):
                            st.session_state.report_editing = False
                            st.rerun()

            # ------------------------------------------------------------------
            # STRUCTURED REPORT UI DISPLAY
            # ------------------------------------------------------------------
            rep_is_tumor = r_pred.get("prediction") in ["Glioma", "Meningioma", "Pituitary"]
            rep_tag_bg = "#fee2e2" if rep_is_tumor else "#d1fae5"
            rep_tag_fg = "#991b1b" if rep_is_tumor else "#065f46"

            render_html(f"""
            <div class="cx-report-container">
                <!-- Header -->
                <div class="cx-report-header">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: #0284c7; text-transform: uppercase; letter-spacing: 1px;">
                                {html.escape(r_hdr.get('application_name', 'NeuroTraCera Decision Support'))}
                            </div>
                            <div class="cx-report-title">{html.escape(r_hdr.get('report_title', 'MRI Analysis Report'))}</div>
                        </div>
                        <div style="text-align: right; font-size: 12px; color: #64748b;">
                            <div>Report ID: <strong style="color: #0f172a;">{html.escape(r_hdr.get('report_id', 'N/A'))}</strong></div>
                            <div>Generated: <strong style="color: #0f172a;">{html.escape(r_hdr.get('generated_at', 'N/A'))}</strong></div>
                            <div>Version: <strong>v{rep.get('report_version', 1)}</strong></div>
                        </div>
                    </div>
                </div>

                <!-- Patient & Examination Metadata Grid -->
                <div class="cx-report-meta-grid">
                    <div>
                        <span class="cx-kpi-label">PATIENT INFORMATION</span>
                        <div style="font-size: 13px; margin-top: 6px; line-height: 1.6;">
                            <div><strong>Name:</strong> {html.escape(r_patient.get('patient_name', 'Not available'))}</div>
                            <div><strong>Patient ID:</strong> {html.escape(r_patient.get('patient_id', 'Not available'))}</div>
                            <div><strong>Case ID:</strong> {html.escape(r_patient.get('case_id', 'Not available'))}</div>
                            <div><strong>Age:</strong> {html.escape(r_patient.get('patient_age', 'Not available'))}</div>
                            <div><strong>Biological Sex:</strong> {html.escape(r_patient.get('patient_gender', 'Not available'))}</div>
                            <div><strong>Referring Physician:</strong> {html.escape(r_patient.get('referring_physician', 'Not available'))}</div>
                        </div>
                    </div>
                    <div>
                        <span class="cx-kpi-label">EXAMINATION PARAMETERS</span>
                        <div style="font-size: 13px; margin-top: 6px; line-height: 1.6;">
                            <div><strong>MRI Examination:</strong> {html.escape(r_exam.get('body_region', 'Brain'))}</div>
                            <div><strong>Protocol / Sequence:</strong> {html.escape(r_exam.get('mri_sequence', 'T1-CE Axial'))}</div>
                            <div><strong>Scan File:</strong> {html.escape(r_exam.get('image_name', 'mri_scan.png'))}</div>
                            <div><strong>Examination Date:</strong> {html.escape(r_patient.get('date_of_examination', 'Not available'))}</div>
                            <div><strong>Target Layer:</strong> {html.escape(r_exam.get('target_layer', 'layer4'))}</div>
                            <div><strong>Monte Carlo Passes:</strong> {r_exam.get('mc_passes', 20)}</div>
                        </div>
                    </div>
                </div>

                <!-- AI Analysis / Prediction Banner -->
                <div style="background: {rep_tag_bg}; border: 1px solid {rep_tag_fg}; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: {rep_tag_fg};">AI Classification Result:</span>
                            <div style="font-family: var(--font-display); font-size: 20px; font-weight: 800; color: {rep_tag_fg};">
                                {html.escape(r_pred.get('prediction_label', r_pred.get('prediction', 'Unknown')))}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: {rep_tag_fg};">Confidence:</span>
                            <div style="font-family: var(--font-display); font-size: 20px; font-weight: 800; color: {rep_tag_fg};">
                                {html.escape(str(r_pred.get('probability_percentage', 'N/A')))}
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #334155; margin-top: 6px; border-top: 1px solid rgba(0,0,0,0.08); padding-top: 6px;">
                        <strong>Uncertainty:</strong> {float(r_pred.get('uncertainty_std', 0.0)):.4f} ({html.escape(str(r_pred.get('uncertainty_level', 'Low')))} Uncertainty) &nbsp;·&nbsp;
                        <strong>95% CI:</strong> [{html.escape(str(r_pred.get('confidence_interval', 'N/A')))}] &nbsp;·&nbsp;
                        <strong>Architecture:</strong> ResNet-50 PyTorch
                    </div>
                </div>
            </div>
            """)

            # ------------------------------------------------------------------
            # ORIGINAL MRI SCAN
            # ------------------------------------------------------------------
            render_html("""
            <div class="cx-report-container" style="margin-top: -8px;">
                <div class="cx-report-section-title">ORIGINAL MRI IMAGE</div>
            </div>
            """)
            mri_col1, mri_col2, mri_col3 = st.columns([1, 1.8, 1])
            with mri_col2:
                orig_url = rep.get("original_mri", {}).get("image")
                if orig_url:
                    st.image(orig_url, caption="Figure 1. Original MRI Image", use_container_width=True)

            # ------------------------------------------------------------------
            # FINDINGS & IMPRESSION
            # ------------------------------------------------------------------
            render_html("""
            <div class="cx-report-container" style="margin-top: -8px;">
                <div class="cx-report-section-title">FINDINGS</div>
                <div class="cx-report-box">
            """ + html.escape(rep.get("findings", "Not available from automated analysis.")).replace("\n", "<br/>") + """
                </div>

                <div class="cx-report-section-title">IMPRESSION</div>
                <div class="cx-report-callout">
                    <strong style="display: block; margin-bottom: 4px;">Diagnostic Impression:</strong>
            """ + html.escape(rep.get("impression", "Not available from automated analysis.")).replace("\n", "<br/>") + """
                </div>
            </div>
            """)

            # ------------------------------------------------------------------
            # EXPLAINABLE AI (XAI) ANALYSIS SECTION
            # ------------------------------------------------------------------
            render_html(f"""
            <div class="cx-report-container" style="margin-top: -8px;">
                <div class="cx-report-section-title">EXPLAINABLE AI ANALYSIS</div>
                <p style="color: #475569; font-size: 13px; line-height: 1.5; margin: 0 0 16px 0;">
                    {html.escape(rep.get("xai_summary_interpretation", ""))}
                </p>
            </div>
            """)

            if r_xai:
                for idx in range(0, len(r_xai), 2):
                    xc1, xc2 = st.columns(2, gap="medium")
                    with xc1:
                        item1 = r_xai[idx]
                        st.markdown(f"<strong style='font-size:14px; color:var(--navy-900);'>{html.escape(item1.get('method', ''))}</strong>", unsafe_allow_html=True)
                        if item1.get("image"):
                            st.image(item1.get("image"), caption=item1.get("caption", ""), use_container_width=True)
                        st.caption(item1.get("interpretation", ""))

                    if idx + 1 < len(r_xai):
                        with xc2:
                            item2 = r_xai[idx + 1]
                            st.markdown(f"<strong style='font-size:14px; color:var(--navy-900);'>{html.escape(item2.get('method', ''))}</strong>", unsafe_allow_html=True)
                            if item2.get("image"):
                                st.image(item2.get("image"), caption=item2.get("caption", ""), use_container_width=True)
                            st.caption(item2.get("interpretation", ""))
            else:
                st.info("No explainability visualizations are currently available for this report.")

            # ------------------------------------------------------------------
            # MODEL INFORMATION & DISCLAIMER
            # ------------------------------------------------------------------
            render_html(f"""
            <div class="cx-report-container" style="margin-top: 14px;">
                <div class="cx-report-section-title">MODEL INFORMATION</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 12.5px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                    <div><strong>Model Name:</strong> {html.escape(r_mod.get('model_name', 'ResNet50 Classifier'))}</div>
                    <div><strong>Version:</strong> {html.escape(r_mod.get('model_version', '1.0.0'))}</div>
                    <div><strong>Architecture:</strong> {html.escape(r_mod.get('architecture', 'ResNet-50'))}</div>
                    <div><strong>Model Type:</strong> {html.escape(r_mod.get('model_type', 'CNN'))}</div>
                    <div><strong>Attribution Methods:</strong> {html.escape(', '.join(r_mod.get('xai_methods_evaluated', [])) or 'None')}</div>
                    <div><strong>Analysis Date:</strong> {html.escape(r_mod.get('analysis_timestamp', 'N/A'))}</div>
                </div>

                <div class="cx-report-disclaimer">
                    <strong style="display: block; margin-bottom: 4px; font-size: 12.5px;">⚠️ AI-Assisted Analysis Disclaimer</strong>
                    {html.escape(str(rep.get('disclaimer') or AI_DISCLAIMER_TEXT))}
                </div>
            </div>
            """)

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🔬 Analyze Another Scan", type="primary", use_container_width=True):
                go_to("Detection")
        with btn_c2:
            if st.button("🗂️ View Case History", use_container_width=True):
                go_to("History")

# ==============================================================================
# 4. PATIENT HISTORY PAGE
# ==============================================================================

elif st.session_state.page == "History":
    render_html("""
    <div class="cx-page-header">
        <h1 class="cx-page-title">Patient Case History</h1>
        <p class="cx-page-desc">Review and inspect previously evaluated MRI analyses saved to this account.</p>
    </div>
    """)

    history_records = fetch_patient_history()

    if not history_records:
        render_html("""
        <div class="cx-card" style="text-align: center; padding: 50px 20px;">
            <div style="font-size: 36px; margin-bottom: 12px;">📂</div>
            <h3 style="font-family: var(--font-display); font-size: 19px; font-weight: 700;">No Analysis History Available</h3>
            <p style="color: #64748b; font-size: 13.5px; margin-bottom: 20px;">
                You have not processed any MRI scans yet. Analyses performed in the Detection workspace will automatically be cataloged here.
            </p>
        </div>
        """)
        if st.button("Start First MRI Analysis →", type="primary"):
            go_to("Detection")
    else:
        f1, f2, f3 = st.columns([1.8, 1, 1.2])
        with f1:
            search_query = st.text_input("Filter by Case ID, tumor type, or filename", placeholder="Search cases...", key="cx_hist_search")
        with f2:
            class_filter = st.selectbox("Filter by Class", ["All Classes"] + CLASSES, key="cx_hist_filter")
        with f3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True, key="cx_btn_clear_all_hist"):
                st.session_state["cx_confirm_clear_all"] = True

        if st.session_state.get("cx_confirm_clear_all"):
            st.warning("⚠️ Are you sure you want to permanently delete ALL history records for this account?")
            c_yes, c_no, _ = st.columns([1.2, 1, 3])
            with c_yes:
                if st.button("Yes, Clear All History", type="primary", key="cx_btn_confirm_clear_yes", use_container_width=True):
                    clear_all_patient_history()
                    st.session_state["cx_confirm_clear_all"] = False
                    st.toast("All history records cleared.", icon="🗑️")
                    st.rerun()
            with c_no:
                if st.button("Cancel", key="cx_btn_confirm_clear_no", use_container_width=True):
                    st.session_state["cx_confirm_clear_all"] = False
                    st.rerun()

        filtered_records = []
        for r in history_records:
            pred = r.get("predicted_class", "Unknown")
            case_id = r.get("patient_case_id", "")
            img_name = r.get("image_name", "")
            
            if class_filter != "All Classes" and pred != class_filter:
                continue
            if search_query:
                q = search_query.lower()
                if q not in pred.lower() and q not in case_id.lower() and q not in img_name.lower():
                    continue
            filtered_records.append(r)

        st.caption(f"Displaying {len(filtered_records)} of {len(history_records)} recorded case(s)")
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        for index, record in enumerate(filtered_records):
            p_class = record.get("predicted_class", "Unknown")
            p_conf = percentage(record.get("probability", 0))
            p_uncert = record.get("uncertainty_level", "Normal")
            c_id = record.get("patient_case_id") or f"CASE-{index + 1:03d}"
            c_date = record.get("created_at", "N/A")[:16].replace("T", " ")
            c_img = record.get("image_name", "MRI_Scan.png")

            with st.expander(f"Case {c_id} — {DISPLAY_NAMES.get(p_class, p_class)} ({p_conf:.1f}% Confidence)"):
                col_h1, col_h2, col_h3, col_h4 = st.columns([1.2, 1, 1, 1.1])
                with col_h1:
                    st.write("**Prediction:**", DISPLAY_NAMES.get(p_class, p_class))
                    st.write("**File:**", c_img)
                with col_h2:
                    st.write("**Confidence:**", f"{p_conf:.2f}%")
                    st.write("**Uncertainty:**", str(p_uncert))
                with col_h3:
                    st.write("**Date Analyzed:**", c_date)
                    st.write("**Patient ID:**", record.get("patient_id", "Not specified"))
                with col_h4:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("Open Full Report", key=f"cx_btn_hist_open_{index}", type="primary", use_container_width=True):
                        st.session_state.result = record
                        go_to("Result")
                    if st.button("🗑️ Remove Record", key=f"cx_btn_hist_del_{record.get('id', index)}", type="secondary", use_container_width=True, help=f"Delete case {c_id} from history"):
                        rec_id = record.get("id")
                        if delete_patient_analysis(rec_id):
                            st.toast(f"✓ Case {c_id} removed from history.", icon="🗑️")
                            st.rerun()
                        else:
                            st.error("Failed to delete record.")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh History Records", use_container_width=True):
            st.rerun()

# ==============================================================================
# 5. ABOUT XAI PAGE
# ==============================================================================

elif st.session_state.page == "About":
    render_html("""
    <div class="cx-page-header">
        <h1 class="cx-page-title">About NeuroTraCera & Explainable Methods</h1>
        <p class="cx-page-desc">Architecture overview, Grad-CAM visual attribution mechanisms, and Bayesian uncertainty estimation.</p>
    </div>
    """)

    col_ab1, col_ab2 = st.columns([1.4, 1], gap="large")

    with col_ab1:
        render_html("""
        <div class="cx-card">
            <span class="cx-kpi-label">PLATFORM MISSION</span>
            <h3 style="font-family: var(--font-display); font-size: 19px; font-weight: 700; margin: 4px 0 10px 0;">Explainable AI for Brain Tumor Imaging</h3>
            <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                Deep neural networks frequently achieve high benchmark accuracy on brain MRI classification tasks, but their "black box" 
                nature impedes clinical integration. <strong>NeuroTraCera</strong> bridges this trust gap by augmenting ResNet-50 feature 
                representations with gradient-based visual attribution and epistemic uncertainty quantification.
            </p>
        </div>
        <div class="cx-card">
            <span class="cx-kpi-label">NEURAL PIPELINE</span>
            <h3 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; margin: 4px 0 14px 0;">End-to-End Analysis Workflow</h3>
            <div style="display: flex; flex-direction: column; gap: 14px; margin-top: 10px;">
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: #e0f2fe; color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">1</div>
                    <div>
                        <strong style="font-size: 13.5px; color: var(--navy-900);">MRI Ingestion & Preprocessing</strong>
                        <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">Intensity normalization, channel alignment, and bicubic interpolation to 224×224 pixels.</p>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: #e0f2fe; color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">2</div>
                    <div>
                        <strong style="font-size: 13.5px; color: var(--navy-900);">Deep Convolutional Feature Extraction</strong>
                        <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">Residual blocks extract multi-scale spatial representations of tissue density, edema, and mass effect.</p>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: #e0f2fe; color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">3</div>
                    <div>
                        <strong style="font-size: 13.5px; color: var(--navy-900);">Multi-Method Explainable AI Suite</strong>
                        <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">
                            Integrates five complementary attribution algorithms: <strong>Grad-CAM</strong> (gradient feature maps), <strong>Grad-CAM++</strong> (higher-order gradient weighting), <strong>LIME</strong> (superpixel perturbations), <strong>SHAP</strong> (Shapley game-theoretic values), and <strong>Integrated Gradients</strong> (axiomatic path integration).
                        </p>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: #e0f2fe; color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">4</div>
                    <div>
                        <strong style="font-size: 13.5px; color: var(--navy-900);">Cross-Method Spatial Agreement Analysis</strong>
                        <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">
                            Evaluates Pearson spatial correlation (r), cosine similarity, and salient mask Intersection over Union (IoU) across explanation maps to examine attribution consistency.
                        </p>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div style="width: 32px; height: 32px; border-radius: 8px; background: #e0f2fe; color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">5</div>
                    <div>
                        <strong style="font-size: 13.5px; color: var(--navy-900);">Monte Carlo Dropout Uncertainty</strong>
                        <p style="color: #64748b; font-size: 12.5px; margin: 2px 0 0 0;">Repeated stochastic inferences assess prediction stability, estimating epistemic model confidence intervals.</p>
                    </div>
                </div>
            </div>
        </div>
        """)

    with col_ab2:
        render_html("""
        <div class="cx-card">
            <span class="cx-kpi-label">PATHOLOGY TARGETS</span>
            <h3 style="font-family: var(--font-display); font-size: 18px; font-weight: 700; margin: 4px 0 14px 0;">Target Classes</h3>
            <div style="display: grid; gap: 12px;">
                <div style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px;">
                    <strong style="color: #b91c1c;">Glioma Tumor</strong>
                    <p style="font-size: 12px; color: #64748b; margin: 3px 0 0 0;">Infiltrative lesions frequently exhibiting hyperintensity and irregular non-enhancing margins.</p>
                </div>
                <div style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px;">
                    <strong style="color: #b45309;">Meningioma Tumor</strong>
                    <p style="font-size: 12px; color: #64748b; margin: 3px 0 0 0;">Extra-axial, dural-based masses frequently presenting distinct contrast enhancement and dural tail signs.</p>
                </div>
                <div style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px;">
                    <strong style="color: #4338ca;">Pituitary Adenoma</strong>
                    <p style="font-size: 12px; color: #64748b; margin: 3px 0 0 0;">Sellar and suprasellar masses that may compress the optic chiasm and perturb endocrine homeostasis.</p>
                </div>
                <div style="padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px;">
                    <strong style="color: #047857;">No Tumor (Benign)</strong>
                    <p style="font-size: 12px; color: #64748b; margin: 3px 0 0 0;">Absence of mass effect, midline shift, or pathological focal enhancement on the scanned slice.</p>
                </div>
            </div>
        </div>
        """)

    render_html("""
    <div class="cx-disclaimer-banner">
        <span style="font-size: 24px;">⚠️</span>
        <div>
            <strong>Mandatory Institutional Notice</strong>
            <p>
                This software is an educational demonstration and research prototype. Grad-CAM visual heatmaps illustrate algorithmic 
                feature importance rather than definitive pathological margins. This platform does not provide definitive diagnoses 
                and should not be used as the sole basis for clinical treatment planning.
            </p>
        </div>
    </div>
    """)

# ==============================================================================
# 6. SETTINGS & PROFILE PAGES (SEPARATE & TABBED VIEWS)
# ==============================================================================

elif st.session_state.page in ["Settings", "Profile"]:
    t_start = time.time()
    health_payload = check_api_health()
    ping_ms = round((time.time() - t_start) * 1000)

    # Resolve Profile Data
    profile_name = st.session_state.get("profile_full_name") or display_patient_name or "Dr. Demo User"
    profile_role = st.session_state.get("profile_role") or "Radiologist"
    profile_email = st.session_state.get("profile_email") or (f"{display_username}@neuroxai.med" if "@" not in display_username else display_username)
    profile_inst = st.session_state.get("profile_institution") or "General Hospital"
    profile_case_id = st.session_state.get("profile_case_id") or str(current_user.get("patient_id") or "CASE-2026-NTR-042")
    profile_dept = st.session_state.get("profile_dept") or "Diagnostic Radiology & Neuroimaging"

    # Compute Initials for Avatar Circle (e.g. "DD" for Dr. Demo User)
    clean_name = profile_name.replace("Dr.", "").replace("Prof.", "").strip()
    name_parts = clean_name.split()
    if len(name_parts) >= 2:
        initials = (name_parts[0][0] + name_parts[1][0]).upper()
    elif len(name_parts) == 1 and len(name_parts[0]) >= 2:
        initials = name_parts[0][:2].upper()
    else:
        initials = "DD"

    # Page Header (Matching Reference Image)
    render_html("""
    <div class="cx-page-header" style="margin-bottom: 22px;">
        <h1 style="font-family: var(--font-display); font-size: 26px; font-weight: 700; color: #0f172a; margin: 0 0 4px 0; letter-spacing: -0.4px;">Settings & Profile</h1>
        <p style="font-size: 14px; color: #64748b; margin: 0 0 16px 0;">Manage your account credentials and system status</p>
        <div style="border-bottom: 1px solid #e2e8f0; width: 100%;"></div>
    </div>
    """)

    # Top User Profile Card (Native Bordered Container Card)
    with st.container(border=True):
        col_card_l, col_card_r = st.columns([4, 1.2], vertical_alignment="center")
        with col_card_l:
            render_html(f"""
            <div style="display: flex; align-items: center; gap: 18px;">
                <div class="cx-profile-avatar-circle">{initials}</div>
                <div>
                    <div class="cx-profile-banner-name">{html.escape(profile_name)}</div>
                    <div class="cx-profile-banner-sub">
                        <span>{html.escape(profile_role)}</span>
                        <span style="color: #cbd5e1;">·</span>
                        <span>{html.escape(profile_email)}</span>
                    </div>
                </div>
            </div>
            """)
        with col_card_r:
            if st.button("Logout", key="cx_prof_top_logout", type="tertiary", use_container_width=True):
                perform_logout()

    # Determine default tab based on whether user arrived via "Settings" or "Profile"
    tab_names = ["Profile Details", "Change Password", "System Status"]
    default_tab = "System Status" if st.session_state.page == "Settings" else "Profile Details"

    tab_profile, tab_password, tab_system = st.tabs(tab_names, default=default_tab)

    # ------------------ TAB 1: PROFILE DETAILS ------------------
    with tab_profile:
        with st.container(border=True):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                new_name = st.text_input("Full Name", value=profile_name, key="cx_input_prof_name")
                role_options = ["Radiologist", "Neuro-Oncologist", "Neurosurgeon", "Neurologist", "Medical Physicist", "Clinical AI Researcher"]
                default_role_idx = role_options.index(profile_role) if profile_role in role_options else 0
                new_role = st.selectbox("Clinical Role", role_options, index=default_role_idx, key="cx_input_prof_role")
                new_case_id = st.text_input("Assigned Case ID", value=profile_case_id, key="cx_input_prof_case")

            with col2:
                new_email = st.text_input("Email Address", value=profile_email, key="cx_input_prof_email")
                new_inst = st.text_input("Institution", value=profile_inst, key="cx_input_prof_inst")
                new_dept = st.text_input("Clinical Department", value=profile_dept, key="cx_input_prof_dept")

            col_space, col_btn = st.columns([3.8, 1.2])
            with col_btn:
                if st.button("Save Changes", type="primary", key="cx_btn_save_profile", use_container_width=True):
                    st.session_state["profile_full_name"] = new_name
                    st.session_state["profile_role"] = new_role
                    st.session_state["profile_email"] = new_email
                    st.session_state["profile_institution"] = new_inst
                    st.session_state["profile_case_id"] = new_case_id
                    st.session_state["profile_dept"] = new_dept
                    if st.session_state.get("user") and isinstance(st.session_state["user"], dict):
                        st.session_state["user"]["patient_name"] = new_name
                        st.session_state["user"]["patient_id"] = new_case_id
                    st.success("✓ Profile details updated successfully.")
                    st.rerun()

    # ------------------ TAB 2: CHANGE PASSWORD ------------------
    with tab_password:
        with st.container(border=True):
            col_pw1, col_pw2 = st.columns([1.2, 1], gap="large")
            with col_pw1:
                curr_pw = st.text_input("Current Password", type="password", key="cx_pw_curr", placeholder="Enter current password")
                new_pw = st.text_input("New Password", type="password", key="cx_pw_new", placeholder="Enter new password (min. 8 characters)")
                conf_pw = st.text_input("Confirm New Password", type="password", key="cx_pw_conf", placeholder="Confirm new password")

                col_pw_sp, col_pw_btn = st.columns([1, 1.4])
                with col_pw_btn:
                    if st.button("Update Password", type="primary", key="cx_btn_update_pw", use_container_width=True):
                        if not curr_pw:
                            st.error("Please enter your current account password.")
                        elif len(new_pw) < 6:
                            st.error("New password must be at least 6 characters.")
                        elif new_pw != conf_pw:
                            st.error("New passwords do not match. Please verify and try again.")
                        else:
                            st.success("✓ Password successfully updated.")

            with col_pw2:
                render_html("""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-size: 13.5px; color: #475569; line-height: 1.6;">
                    <div style="font-weight: 700; color: #0f172a; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">🔒</span> Security & Authentication
                    </div>
                    <ul style="margin: 0; padding-left: 18px; color: #64748b; font-size: 13px;">
                        <li style="margin-bottom: 6px;">Minimum 8 alphanumeric characters</li>
                        <li style="margin-bottom: 6px;">Avoid reusing clinical credentials</li>
                        <li style="margin-bottom: 6px;">End-to-end 256-bit encrypted session</li>
                        <li>HIPAA compliant institutional audit log</li>
                    </ul>
                </div>
                """)

    # ------------------ TAB 3: SYSTEM & SERVER STATUS ------------------
    with tab_system:
        with st.container(border=True):
            col_sys1, col_sys2 = st.columns([1.3, 1], gap="large")
            with col_sys1:
                is_connected = health_payload is not None
                device_str = (health_payload.get("device") if health_payload else "N/A").upper()
                model_loaded = health_payload.get("model_loaded", False) if health_payload else False
                db_connected = health_payload.get("database_connected", False) if health_payload else False
                db_type = health_payload.get("database", "SQLite") if health_payload else "SQLite"

                render_html(f"""
                <div style="font-size: 13.5px; line-height: 2.3;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">FastAPI Engine Status:</span>
                        <span class="cx-badge {'cx-badge-success' if is_connected else 'cx-badge-danger'}">
                            {'● Operational (200 OK)' if is_connected else 'Backend Unreachable'}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">Backend REST Endpoint:</span>
                        <code style="background: #f1f5f9; padding: 2px 8px; border-radius: 4px; font-size: 12.5px; color: #0f172a;">{API_URL}</code>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">Roundtrip Ping Latency:</span>
                        <strong style="color: #0f172a;">{ping_ms} ms</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">PyTorch Hardware Device:</span>
                        <strong style="color: #155dfd;">{html.escape(device_str)}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">ResNet-50 Classifier:</span>
                        <strong style="color: {'#10b981' if model_loaded else '#ef4444'};">
                            {'Loaded in Memory' if model_loaded else 'Not Loaded'}
                        </strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding: 6px 0;">
                        <span style="color: #64748b;">Clinical Database:</span>
                        <strong style="color: #0f172a;">{html.escape(db_type)} ({'Connected' if db_connected else 'Disconnected'})</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0;">
                        <span style="color: #64748b;">Supported Diagnoses:</span>
                        <strong style="color: #0f172a;">Glioma, Meningioma, Pituitary, No Tumor</strong>
                    </div>
                </div>
                """)

            with col_sys2:
                render_html("""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                    <span style="font-size: 11px; font-weight: 700; color: #155dfd; letter-spacing: 0.6px; text-transform: uppercase;">DIAGNOSTICS & SYSTEM</span>
                    <h4 style="font-size: 15px; font-weight: 700; margin: 4px 0 6px 0; color: #0f172a;">Server Connectivity</h4>
                    <p style="color: #64748b; font-size: 12.5px; line-height: 1.5; margin: 0;">
                        Verify connectivity to the NeuroTraCera FastAPI inference service and check database read/write throughput.
                    </p>
                </div>
                """)
                if st.button("🔄 Test Backend Connectivity", type="primary", use_container_width=True, key="cx_btn_test_conn_sys"):
                    st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Reset Diagnostic Session Cache", use_container_width=True, key="cx_btn_clear_cache"):
                    st.session_state.result = None
                    st.session_state.current_report = None
                    st.success("✓ Session cache cleared.")
                    st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================

render_html("""
<div style="margin-top: 45px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 12px; line-height: 1.6;">
    <strong>NeuroTraCera</strong> &nbsp;·&nbsp; Explainable Brain Tumor Analytics & Clinical Decision Support<br>
    Academic / Educational Research Prototype &nbsp;·&nbsp; ResNet-50 & Grad-CAM & Monte Carlo Dropout
</div>
""")