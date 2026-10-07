import base64
import uuid
import io
import hashlib
import secrets

from datetime import datetime, timezone

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Header,
    Response
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from analysis import (
    AnalysisRequest,
    XAIExecutionRequest,
    ReportGenerationRequest,
    ReportUpdateRequest,
    ReportResponse,
    PDFGenerationRequest
)
from model import ModelEngine, CLASSES
import xai_engine
from report_generator import (
    build_mri_report,
    generate_report_pdf,
    format_report_text
)

from gradcam import (
    GradCAM,
    get_target_layer,
    heatmap_and_overlay
)

from sampledata import (
    get_all_samples_manifest,
    get_sample_mri_by_id
)

from db import Database

from evaluation import uncertainty_level
from dashboard import build_dashboard


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="XAI Brain Tumor Detection & Localization API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL + DATABASE
# ============================================================

engine = ModelEngine()
database = Database()


# ============================================================
# SESSION STORAGE
# ============================================================

sessions = {}


# ============================================================
# AUTHENTICATION MODELS
# ============================================================

class RegisterRequest(BaseModel):
    username: str
    password: str
    patient_name: str | None = None
    patient_id: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str | None = None
    email: str | None = None
    name: str | None = None
    google_id: str | None = None


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password: str):

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


def verify_password(
    password: str,
    stored_hash: str
):

    try:

        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)

        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000
        )

        return secrets.compare_digest(
            actual_hash,
            expected_hash
        )

    except Exception:

        return False


# ============================================================
# SESSION TOKEN
# ============================================================

def generate_session_token():

    return secrets.token_urlsafe(32)


async def get_current_user(
    session_token: str | None
):

    if not session_token:

        raise HTTPException(
            status_code=401,
            detail="Please login first."
        )

    user_id = sessions.get(
        session_token
    )

    if not user_id:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session."
        )

    user = await database.get_user_by_id(
        user_id
    )

    if not user:

        sessions.pop(
            session_token,
            None
        )

        raise HTTPException(
            status_code=401,
            detail="User not found."
        )

    return user


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():

    connected = await database.connect()

    if connected:

        print(
            "DATABASE CONNECTED SUCCESSFULLY"
        )

        print(
            f"SQLite DB: {database.db_path}"
        )

    else:

        print(
            "WARNING: DATABASE NOT CONNECTED"
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "XAI Brain Tumor API is running",
        "docs": "/docs",
        "database": "SQLite"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "model_loaded": engine.model_loaded,
        "device": str(engine.device),
        "classes": CLASSES,
        "database": "SQLite",
        "database_connected": True
    }


@app.get("/landing")
async def get_landing_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "streamlit", "landing_bundle.html")
    if os.path.exists(bundle_path):
        return FileResponse(bundle_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Landing page bundle not found")


# ============================================================
# REGISTER
# ============================================================

@app.post("/api/register")
async def register(
    request: RegisterRequest
):

    username = request.username.strip()

    password = request.password

    if len(username) < 3:

        raise HTTPException(
            status_code=400,
            detail="Username must contain at least 3 characters."
        )

    if len(password) < 6:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters."
        )

    existing_user = await database.get_user_by_username(
        username
    )

    if existing_user:

        raise HTTPException(
            status_code=409,
            detail="Username already exists."
        )

    password_hash = hash_password(
        password
    )

    user_id = await database.create_user(
        username=username,
        password_hash=password_hash,
        patient_name=request.patient_name,
        patient_id=request.patient_id
    )

    if not user_id:

        raise HTTPException(
            status_code=500,
            detail="Unable to create account."
        )

    return {
        "success": True,
        "message": "Account created successfully.",
        "user_id": user_id,
        "username": username
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/login")
async def login(
    request: LoginRequest
):

    username = request.username.strip()

    user = await database.get_user_by_username(
        username
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    if not verify_password(
        request.password,
        user.get("password_hash", "")
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    session_token = generate_session_token()

    sessions[session_token] = str(
        user["id"]
    )

    await database.update_last_login(
        str(user["id"])
    )

    return {
        "success": True,
        "message": "Login successful.",
        "session_token": session_token,

        "user": {
            "id": str(user["id"]),

            "username": user.get(
                "login_id",
                ""
            ),

            "patient_name": user.get(
                "patient_name"
            ),

            "patient_id": user.get(
                "patient_id"
            )
        }
    }


# ============================================================
# GOOGLE AUTHENTICATION
# ============================================================

@app.post("/api/auth/google")
async def google_auth(
    request: GoogleAuthRequest
):
    email = None
    name = None
    sub = None

    # 1. If an ID token was provided, verify with Google tokeninfo
    if request.id_token and request.id_token.strip():
        try:
            import requests as req
            resp = req.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={request.id_token.strip()}",
                timeout=6
            )
            if resp.status_code == 200:
                payload = resp.json()
                email = payload.get("email")
                name = payload.get("name") or payload.get("given_name")
                sub = payload.get("sub")
            else:
                print(f"Google tokeninfo response {resp.status_code}: {resp.text}")
        except Exception as err:
            print(f"Google OAuth verification request failed: {err}")

    # 2. Support direct verified payload or demo/sandbox profile
    if not email and request.email and request.email.strip():
        email = request.email.strip().lower()
        name = request.name or f"Dr. {email.split('@')[0].capitalize()}"
        sub = request.google_id or f"goog_{secrets.token_hex(4)}"

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Valid Google authentication token or email required."
        )

    # 3. Lookup user or register automatically
    user = await database.get_user_by_username(email)
    if not user:
        random_pwd = secrets.token_urlsafe(24)
        pwd_hash = hash_password(random_pwd)
        patient_tag = f"GOOG-{sub[-6:] if sub and len(sub) >= 6 else '001'}"
        user_id = await database.create_user(
            username=email,
            password_hash=pwd_hash,
            patient_name=name or "Clinical Specialist",
            patient_id=patient_tag
        )
        if not user_id:
            raise HTTPException(
                status_code=500,
                detail="Unable to initialize Google clinical user profile."
            )
        user = await database.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve clinical account profile."
        )

    session_token = generate_session_token()
    sessions[session_token] = str(user["id"])
    await database.update_last_login(str(user["id"]))

    return {
        "success": True,
        "message": "Google authentication successful.",
        "session_token": session_token,
        "user": {
            "id": str(user["id"]),
            "username": user.get("login_id", email),
            "patient_name": user.get("patient_name") or name or "Clinical Specialist",
            "patient_id": user.get("patient_id", "")
        }
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/api/logout")
async def logout(
    x_session_token: str | None = Header(
        default=None
    )
):

    if x_session_token:

        sessions.pop(
            x_session_token,
            None
        )

    return {
        "success": True,
        "message": "Logged out successfully."
    }


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/api/me")
async def current_user(
    x_session_token: str | None = Header(
        default=None
    )
):

    user = await get_current_user(
        x_session_token
    )

    return {
        "success": True,

        "user": {
            "id": str(user["id"]),

            "username": user.get(
                "login_id",
                ""
            ),

            "patient_name": user.get(
                "patient_name"
            ),

            "patient_id": user.get(
                "patient_id"
            )
        }
    }


# ============================================================
# SAMPLE MRI LIST
# ============================================================

@app.get("/api/samples")
async def samples():

    return get_all_samples_manifest()


# ============================================================
# GET SINGLE SAMPLE MRI
# ============================================================

@app.get("/api/samples/{sample_id}")
async def sample(
    sample_id: str
):

    result = get_sample_mri_by_id(
        sample_id
    )

    if not result:

        raise HTTPException(
            status_code=404,
            detail="Sample not found"
        )

    return result


# ============================================================
# BASE64 DECODER
# ============================================================

def _decode(data):

    return engine.decode_base64(
        data
    )


# ============================================================
# IMAGE TO DATA URL
# ============================================================

def _to_data_url(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# MAIN MRI ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze(
    request: AnalysisRequest,
    x_session_token: str | None = Header(
        default=None
    )
):

    # --------------------------------------------------------
    # REQUIRE LOGIN
    # --------------------------------------------------------

    user = await get_current_user(
        x_session_token
    )

    user_id = str(
        user["id"]
    )


    # --------------------------------------------------------
    # GET IMAGE FROM SAMPLE ID
    # --------------------------------------------------------

    if request.sample_id:

        sample_data = get_sample_mri_by_id(
            request.sample_id
        )

        if not sample_data:

            raise HTTPException(
                status_code=404,
                detail="Sample not found"
            )

        image = _decode(
            sample_data["image_url"]
        )

        image_name = (
            request.image_name
            or request.sample_id + ".png"
        )


    # --------------------------------------------------------
    # GET IMAGE FROM BASE64
    # --------------------------------------------------------

    elif request.image_base64:

        image = _decode(
            request.image_base64
        )

        image_name = (
            request.image_name
            or "mri.png"
        )


    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    else:

        raise HTTPException(
            status_code=400,
            detail="Provide image_base64 or sample_id"
        )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    pred, mean, std, mc = engine.mc_predict(
        image,
        request.mc_passes
    )

    probability = float(
        mean[pred]
    )

    uncertainty = float(
        std[pred]
    )


    # ========================================================
    # CONFIDENCE INTERVAL
    # ========================================================

    confidence_interval = (
        max(
            0.0,
            probability - 1.96 * uncertainty
        ),

        min(
            1.0,
            probability + 1.96 * uncertainty
        )
    )


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    x = engine.prepare(
        image
    )


    # ========================================================
    # XAI ATTRIBUTION PIPELINE
    # ========================================================

    target_layer_name = getattr(request, "cam_layer", "layer4") or "layer4"
    alpha_val = getattr(request, "alpha", 0.55) or 0.55
    cmap_val = getattr(request, "colormap", "jet") or "jet"

    raw_maps = {}

    # 1. Grad-CAM
    try:
        gcam_data = xai_engine.generate_gradcam(
            engine.model,
            image,
            engine.transform,
            engine.device,
            pred,
            target_layer_name,
            alpha=alpha_val,
            colormap=cmap_val
        )
        raw_maps["Grad-CAM"] = gcam_data.pop("raw_map")
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gcam_data = {
            "heatmap_url": "",
            "overlay_url": "",
            "bounding_box": None,
            "metrics": {},
            "error": True,
            "message": "Grad-CAM could not be generated for this analysis."
        }

    # 2. Grad-CAM++
    try:
        gcpp_data = xai_engine.generate_gradcam_plus_plus(
            engine.model,
            image,
            engine.transform,
            engine.device,
            pred,
            target_layer_name,
            alpha=alpha_val,
            colormap=cmap_val
        )
        raw_maps["Grad-CAM++"] = gcpp_data.pop("raw_map")
    except Exception as e:
        print(f"Grad-CAM++ error: {e}")
        gcpp_data = {
            "heatmap_url": "",
            "overlay_url": "",
            "bounding_box": None,
            "metrics": {},
            "error": True,
            "message": "Grad-CAM++ could not be generated for this analysis."
        }

    lime_data = None
    shap_data = None
    ig_data = None

    # Full XAI if requested
    if getattr(request, "run_all_xai", False):
        try:
            lime_res = xai_engine.generate_lime(
                engine.model,
                image,
                engine.transform,
                engine.device,
                pred,
                num_samples=50,
                n_segments=35,
                alpha=alpha_val,
                colormap=cmap_val
            )
            raw_maps["LIME"] = lime_res.pop("raw_map")
            lime_data = lime_res
        except Exception as e:
            print(f"LIME error: {e}")
            lime_data = {"error": True, "message": "LIME could not be generated for this analysis."}

        try:
            shap_res = xai_engine.generate_shap(
                engine.model,
                image,
                engine.transform,
                engine.device,
                pred,
                num_samples=50,
                n_segments=35,
                alpha=alpha_val,
                colormap=cmap_val
            )
            raw_maps["SHAP"] = shap_res.pop("raw_map")
            shap_data = shap_res
        except Exception as e:
            print(f"SHAP error: {e}")
            shap_data = {"error": True, "message": "SHAP could not be generated for this analysis."}

        try:
            ig_res = xai_engine.generate_integrated_gradients(
                engine.model,
                image,
                engine.transform,
                engine.device,
                pred,
                n_steps=15,
                alpha=alpha_val,
                colormap=cmap_val
            )
            raw_maps["Integrated Gradients"] = ig_res.pop("raw_map")
            ig_data = ig_res
        except Exception as e:
            print(f"Integrated Gradients error: {e}")
            ig_data = {"error": True, "message": "Integrated Gradients could not be generated for this analysis."}

    cross_metrics = xai_engine.compute_cross_method_metrics(raw_maps) if len(raw_maps) >= 2 else None

    # Backward compatibility with existing frontend keys
    heat_url = gcam_data.get("heatmap_url", "")
    overlay_url = gcam_data.get("overlay_url", "")
    bbox = gcam_data.get("bounding_box")
    metrics = gcam_data.get("metrics", {})

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "id": str(
            uuid.uuid4()
        ),

        "user_id": user_id,

        "username": user.get(
            "login_id",
            ""
        ),

        "patient_name": user.get(
            "patient_name"
        ),

        "patient_id": user.get(
            "patient_id"
        ),

        "image_name": image_name,

        "prediction": CLASSES[pred],

        "predicted_class": CLASSES[pred],

        "class_index": pred,

        "probability": probability,

        "class_probabilities": {
            CLASSES[i]: float(
                mean[i]
            )
            for i in range(
                len(CLASSES)
            )
        },

        "uncertainty": uncertainty,

        "uncertainty_level": uncertainty_level(
            uncertainty
        ),

        "confidence_interval": confidence_interval,

        "mc_passes_count": request.mc_passes,

        "original_image_url": _to_data_url(
            image
        ),

        "gradcam_heatmap_url": heat_url,

        "gradcam_overlay_url": overlay_url,

        "localization": {

            "bounding_box": bbox,

            "available": bbox is not None,

            "method": (
                "Grad-CAM activation region"
            )
        },

        "xai_metrics": metrics,

        "gradcam": gcam_data,

        "gradcam_plus_plus": gcpp_data,

        "lime": lime_data,

        "shap": shap_data,

        "integrated_gradients": ig_data,

        "cross_method_analysis": cross_metrics,

        "interpretation": (
            f"The model assigns highest probability to {CLASSES[pred]} "
            f"({probability * 100:.1f}% confidence). Highlighted regions show "
            "image areas contributing most strongly to this prediction."
        ),

        "sequence": request.sequence,

        "anatomy": request.anatomy,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }


    # ========================================================
    # SAVE RESULT TO SQLITE
    # ========================================================

    await database.insert_analysis(
        result,
        user_id=user_id,
        patient_information={
            "case_id": user.get(
                "patient_id"
            ),
            "name": user.get(
                "patient_name"
            )
        }
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return result


# ============================================================
# RUN DEDICATED / ON-DEMAND XAI METHOD
# ============================================================

@app.post("/api/xai/run")
async def run_xai(
    request: XAIExecutionRequest,
    x_session_token: str | None = Header(default=None)
):
    user = None
    if x_session_token:
        try:
            user = await get_current_user(x_session_token)
        except Exception:
            pass

    if request.sample_id:
        sample_data = get_sample_mri_by_id(request.sample_id)
        if not sample_data:
            raise HTTPException(status_code=404, detail="Sample not found")
        image = _decode(sample_data["image_url"])
    elif request.image_base64:
        image = _decode(request.image_base64)
    else:
        raise HTTPException(status_code=400, detail="Provide image_base64 or sample_id")

    # Prediction class index
    if request.prediction_class_index is not None:
        pred = int(request.prediction_class_index)
    else:
        pred, _ = engine.predict(image)

    target_layer_name = request.cam_layer or "layer4"
    alpha_val = float(request.alpha or 0.55)
    cmap_val = str(request.colormap or "jet")

    requested_method = str(request.method or "all").lower().strip()
    results = {
        "prediction_class": CLASSES[pred],
        "class_index": pred,
        "requested_method": requested_method,
    }
    raw_maps = {}

    # 1. Grad-CAM
    if requested_method in ["gradcam", "all"]:
        try:
            gcam = xai_engine.generate_gradcam(
                engine.model, image, engine.transform, engine.device,
                pred, target_layer_name, alpha=alpha_val, colormap=cmap_val
            )
            raw_maps["Grad-CAM"] = gcam.pop("raw_map")
            results["gradcam"] = gcam
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            results["gradcam"] = {"error": True, "message": "Grad-CAM could not be generated for this analysis."}

    # 2. Grad-CAM++
    if requested_method in ["gradcam_plus_plus", "gradcam++", "all"]:
        try:
            gcpp = xai_engine.generate_gradcam_plus_plus(
                engine.model, image, engine.transform, engine.device,
                pred, target_layer_name, alpha=alpha_val, colormap=cmap_val
            )
            raw_maps["Grad-CAM++"] = gcpp.pop("raw_map")
            results["gradcam_plus_plus"] = gcpp
        except Exception as e:
            print(f"Grad-CAM++ error: {e}")
            results["gradcam_plus_plus"] = {"error": True, "message": "Grad-CAM++ could not be generated for this analysis."}

    # 3. LIME
    if requested_method in ["lime", "all"]:
        try:
            lime_res = xai_engine.generate_lime(
                engine.model, image, engine.transform, engine.device,
                pred, num_samples=50, n_segments=35, alpha=alpha_val, colormap=cmap_val
            )
            raw_maps["LIME"] = lime_res.pop("raw_map")
            results["lime"] = lime_res
        except Exception as e:
            print(f"LIME error: {e}")
            results["lime"] = {"error": True, "message": "LIME could not be generated for this analysis."}

    # 4. SHAP
    if requested_method in ["shap", "all"]:
        try:
            shap_res = xai_engine.generate_shap(
                engine.model, image, engine.transform, engine.device,
                pred, num_samples=50, n_segments=35, alpha=alpha_val, colormap=cmap_val
            )
            raw_maps["SHAP"] = shap_res.pop("raw_map")
            results["shap"] = shap_res
        except Exception as e:
            print(f"SHAP error: {e}")
            results["shap"] = {"error": True, "message": "SHAP could not be generated for this analysis."}

    # 5. Integrated Gradients
    if requested_method in ["integrated_gradients", "ig", "all"]:
        try:
            ig_res = xai_engine.generate_integrated_gradients(
                engine.model, image, engine.transform, engine.device,
                pred, n_steps=15, alpha=alpha_val, colormap=cmap_val
            )
            raw_maps["Integrated Gradients"] = ig_res.pop("raw_map")
            results["integrated_gradients"] = ig_res
        except Exception as e:
            print(f"Integrated Gradients error: {e}")
            results["integrated_gradients"] = {"error": True, "message": "Integrated Gradients could not be generated for this analysis."}

    if len(raw_maps) >= 2:
        results["cross_method_analysis"] = xai_engine.compute_cross_method_metrics(raw_maps)

    # If analysis_id provided, persist to SQLite
    if request.analysis_id:
        record = await database.get_analysis_by_id(request.analysis_id)
        if record:
            for k, v in results.items():
                if k != "requested_method":
                    record[k] = v
            await database.update_analysis_document(request.analysis_id, record)

    return results


# ============================================================
# UPLOAD MRI IMAGE
# ============================================================

@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    x_session_token: str | None = Header(
        default=None
    )
):

    await get_current_user(
        x_session_token
    )

    data = await file.read()

    try:

        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image"
        )

    encoded = _to_data_url(
        image
    )

    request = AnalysisRequest(
        image_base64=encoded,
        image_name=file.filename
        or "mri.png"
    )

    return await analyze(
        request,
        x_session_token
    )


# ============================================================
# USER-SPECIFIC ANALYSIS HISTORY
# ============================================================

@app.get("/api/analyses")
async def analyses(
    x_session_token: str | None = Header(
        default=None
    )
):

    user = await get_current_user(
        x_session_token
    )

    return await database.user_analyses(
        str(user["id"])
    )


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis_endpoint(
    analysis_id: str,
    x_session_token: str | None = Header(
        default=None
    )
):
    user = await get_current_user(
        x_session_token
    )

    success = await database.delete_analysis(
        analysis_id,
        str(user["id"])
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found or already deleted."
        )

    return {"status": "success", "message": f"Analysis record {analysis_id} removed successfully."}


@app.delete("/api/analyses")
async def clear_all_analyses_endpoint(
    x_session_token: str | None = Header(
        default=None
    )
):
    user = await get_current_user(
        x_session_token
    )

    deleted_count = await database.delete_all_user_analyses(
        str(user["id"])
    )

    return {"status": "success", "message": f"Successfully cleared {deleted_count} analysis record(s).", "count": deleted_count}


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
async def dashboard(
    x_session_token: str | None = Header(
        default=None
    )
):

    user = await get_current_user(
        x_session_token
    )

    records = await database.user_analyses(
        str(user["id"]),
        100
    )

    return build_dashboard(
        records
    )


# ============================================================
# ENSURE ALL 5 XAI METHODS FOR REPORT
# ============================================================

def ensure_all_xai_for_analysis(analysis: dict) -> dict:
    """
    Ensures that an analysis object contains all 5 XAI visualizations
    (Grad-CAM, Grad-CAM++, LIME, SHAP, Integrated Gradients).
    If any are missing and original_image_url is present, computes them.
    """
    orig_url = analysis.get("original_image_url")
    if not orig_url:
        return analysis

    has_gcam = bool(analysis.get("gradcam_overlay_url") or (isinstance(analysis.get("gradcam"), dict) and analysis["gradcam"].get("overlay_url") and not analysis["gradcam"].get("error")))
    has_gcpp = bool(isinstance(analysis.get("gradcam_plus_plus"), dict) and analysis["gradcam_plus_plus"].get("overlay_url") and not analysis["gradcam_plus_plus"].get("error"))
    has_lime = bool(isinstance(analysis.get("lime"), dict) and analysis["lime"].get("overlay_url") and not analysis["lime"].get("error"))
    has_shap = bool(isinstance(analysis.get("shap"), dict) and analysis["shap"].get("overlay_url") and not analysis["shap"].get("error"))
    has_ig = bool(isinstance(analysis.get("integrated_gradients"), dict) and analysis["integrated_gradients"].get("overlay_url") and not analysis["integrated_gradients"].get("error"))

    if has_gcam and has_gcpp and has_lime and has_shap and has_ig:
        return analysis

    try:
        image = _decode(orig_url)
        pred = int(analysis.get("class_index", 0))
        cam_layer = analysis.get("cam_layer", "layer4")
        alpha = float(analysis.get("alpha", 0.55))
        cmap = str(analysis.get("colormap", "jet"))

        if not has_gcam:
            try:
                gcam = xai_engine.generate_gradcam(engine.model, image, engine.transform, engine.device, pred, cam_layer, alpha=alpha, colormap=cmap)
                gcam.pop("raw_map", None)
                analysis["gradcam"] = gcam
                analysis["gradcam_overlay_url"] = gcam.get("overlay_url")
            except Exception as e:
                print(f"Auto Grad-CAM error: {e}")

        if not has_gcpp:
            try:
                gcpp = xai_engine.generate_gradcam_plus_plus(engine.model, image, engine.transform, engine.device, pred, cam_layer, alpha=alpha, colormap=cmap)
                gcpp.pop("raw_map", None)
                analysis["gradcam_plus_plus"] = gcpp
            except Exception as e:
                print(f"Auto Grad-CAM++ error: {e}")

        if not has_lime:
            try:
                lime_res = xai_engine.generate_lime(engine.model, image, engine.transform, engine.device, pred, num_samples=50, n_segments=35, alpha=alpha, colormap=cmap)
                lime_res.pop("raw_map", None)
                analysis["lime"] = lime_res
            except Exception as e:
                print(f"Auto LIME error: {e}")

        if not has_shap:
            try:
                shap_res = xai_engine.generate_shap(engine.model, image, engine.transform, engine.device, pred, num_samples=50, n_segments=35, alpha=alpha, colormap=cmap)
                shap_res.pop("raw_map", None)
                analysis["shap"] = shap_res
            except Exception as e:
                print(f"Auto SHAP error: {e}")

        if not has_ig:
            try:
                ig_res = xai_engine.generate_integrated_gradients(engine.model, image, engine.transform, engine.device, pred, n_steps=15, alpha=alpha, colormap=cmap)
                ig_res.pop("raw_map", None)
                analysis["integrated_gradients"] = ig_res
            except Exception as e:
                print(f"Auto Integrated Gradients error: {e}")

    except Exception as e:
        print(f"ensure_all_xai_for_analysis error: {e}")

    return analysis


# ============================================================
# GENERATE MRI REPORT
# ============================================================

@app.post("/api/generate-mri-report")
@app.post("/generate-mri-report")
async def generate_mri_report_endpoint(
    request: ReportGenerationRequest,
    x_session_token: str | None = Header(default=None)
):
    user_id = None
    if x_session_token:
        try:
            user = await get_current_user(x_session_token)
            user_id = str(user["id"])
        except Exception:
            pass

    analysis = None
    # 1. Try to load from database if analysis_id provided
    if request.analysis_id:
        analysis = await database.get_analysis_by_id(request.analysis_id)

    # 2. Or fallback to analysis_data provided directly in request body
    if not analysis and request.analysis_data:
        analysis = request.analysis_data

    if not analysis:
        raise HTTPException(
            status_code=400,
            detail="MRI analysis result is required before generating a report."
        )

    try:
        analysis = ensure_all_xai_for_analysis(analysis)
        report = build_mri_report(
            analysis=analysis,
            patient_info=request.patient_information,
            examination_info=request.examination_information,
            model_info=request.model_information,
            notes=request.notes,
            user_id=user_id
        )

        # Persist report to database
        await database.insert_report(report)

        return {
            "success": True,
            "message": "Report generated successfully.",
            "report": report
        }
    except Exception as e:
        print(f"Report generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate the report. Please try again."
        )


# ============================================================
# GENERATE MRI REPORT PDF
# ============================================================

@app.post("/api/generate-mri-report/pdf")
@app.post("/generate-mri-report/pdf")
async def generate_mri_report_pdf_endpoint(
    request: PDFGenerationRequest,
    x_session_token: str | None = Header(default=None)
):
    try:
        report_data = request.report
        if not report_data:
            analysis = None
            if request.analysis_id:
                analysis = await database.get_analysis_by_id(request.analysis_id)
            if not analysis and request.analysis_data:
                analysis = request.analysis_data

            if not analysis:
                raise HTTPException(
                    status_code=400,
                    detail="Report data or valid analysis is required to generate PDF."
                )
            analysis = ensure_all_xai_for_analysis(analysis)
            report_data = build_mri_report(analysis)

        pdf_bytes = generate_report_pdf(report_data)
        rep_id = report_data.get("id", "mri_report")
        filename = f"{rep_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        print(f"PDF generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate report PDF. Please try again."
        )


# ============================================================
# GET REPORT BY ID
# ============================================================

@app.get("/api/reports/{report_id}")
async def get_report_endpoint(
    report_id: str,
    x_session_token: str | None = Header(default=None)
):
    report = await database.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"success": True, "report": report}


# ============================================================
# UPDATE REPORT
# ============================================================

@app.put("/api/reports/{report_id}")
async def update_report_endpoint(
    report_id: str,
    request: ReportUpdateRequest,
    x_session_token: str | None = Header(default=None)
):
    update_data = {k: v for k, v in request.dict().items() if v is not None}
    success = await database.update_report(report_id, update_data)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found or update failed")
    updated = await database.get_report_by_id(report_id)
    return {"success": True, "report": updated}