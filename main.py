import os, io, uuid, tempfile
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import trimesh
import numpy as np
from PIL import Image
from rembg import remove
import cv2, svgwrite


load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://refiq-server-bk2cmyue4-alis-projects-48a6b838.vercel.app",
        "https://*.vercel.app"
    ],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BACKEND_SECRET = os.environ["BACKEND_SECRET"]
BUCKET_RAW = os.environ.get("BUCKET_RAW", "raw-photos")
BUCKET_MODELS = os.environ.get("BUCKET_MODELS", "models")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# A simple 3D template (millimeter-ish scale). Tune if needed.
# Order: noseTip, chin, leftEyeOuter, rightEyeOuter, mouthLeft, mouthRight
MODEL_POINTS = np.array([
    [  0.0,   0.0,   0.0],    # nose tip
    [  0.0, -63.6, -12.5],    # chin
    [-43.3,  32.7, -26.0],    # left eye outer
    [ 43.3,  32.7, -26.0],    # right eye outer
    [-28.9, -28.9, -24.1],    # left mouth
    [ 28.9, -28.9, -24.1],    # right mouth
], dtype=np.float32)

class FitPayload(BaseModel):
    job_id: str

def rotationMatrixToEulerAngles(R):
    # yaw (y), pitch (x), roll (z) in degrees
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = 0
    return float(yaw), float(pitch), float(roll)

def exposure_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    # dark/bright fractions by thresholding
    dark_frac = float((gray < 40).sum() / gray.size)
    bright_frac = float((gray > 230).sum() / gray.size)
    return mean, dark_frac, bright_frac

def blur_variance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def auth_or_403(x_api_key: str | None):
#     if x_api_key != BACKEND_SECRET:
#         raise HTTPException(status_code=403, detail="Forbidden")

def _download_bytes(path: str) -> bytes:
    res = supabase.storage.from_(BUCKET_RAW).download(path)
    if getattr(res, "error", None):
        raise RuntimeError(res.error.message)
    return res

def _upload_public(bucket: str, path: str, data: bytes, content_type: str):
    up = supabase.storage.from_(bucket).upload(path, data, {"contentType": content_type, "upsert": "true"})
    if hasattr(up, 'error') and up.error:
        raise RuntimeError(str(up.error))
    return path

def _largest_contour(mask_np: np.ndarray) -> np.ndarray | None:
    # mask_np: uint8 0/255
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]

def _simplify_contour(cnt: np.ndarray, epsilon_ratio: float = 0.003) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)

def _contour_to_svg_path(cnt: np.ndarray, w: int, h: int) -> bytes:
    # Normalize to [0,1] for viewBox
    pts = cnt.reshape(-1, 2).astype(np.float32)
    d = "M " + " L ".join([f"{x/w:.6f} {y/h:.6f}" for (x,y) in pts]) + " Z"
    dwg = svgwrite.Drawing(size=("100%", "100%"), viewBox="0 0 1 1")
    dwg.add(dwg.path(d=d, fill="black", stroke="none"))
    return dwg.tostring().encode("utf-8")

@app.post("/preprocess")
def preprocess(payload: dict = Body(...), x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # if x_api_key != BACKEND_SECRET:
        # raise HTTPException(403, "Forbidden")
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")

    # get job
    job_q = supabase.table("recon_jobs").select("*").eq("id", job_id).single().execute()
    job = job_q.data
    if not job: raise HTTPException(404, "job not found")

    # download front
    front_bytes = _download_bytes(job["front_path"])
    img = Image.open(io.BytesIO(front_bytes)).convert("RGBA")

    # background removal
    cut = remove(img)  # RGBA
    # mask = alpha channel
    alpha = np.array(cut.split()[-1])  # 0..255
    # clean mask (morphology to remove wisps)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    # binarize
    _, mask_bin = cv2.threshold(mask_clean, 0, 255, cv2.THRESH_BINARY)

    # largest contour → simplified polygon
    cnt = _largest_contour(mask_bin)
    if cnt is None:
        raise HTTPException(422, "No subject found in image.")
    cnt_simpl = _simplify_contour(cnt, epsilon_ratio=0.004)

    w, h = cut.size
    svg_bytes = _contour_to_svg_path(cnt_simpl, w, h)

    # encode artifacts
    buf_rgba = io.BytesIO(); cut.save(buf_rgba, format="PNG"); rgba_bytes = buf_rgba.getvalue()
    mask_png = Image.fromarray(mask_bin)
    buf_mask = io.BytesIO(); mask_png.save(buf_mask, format="PNG"); mask_bytes = buf_mask.getvalue()

    base = f"{job['user_id'] or 'anon'}/{job_id}"
    rgba_path = _upload_public(BUCKET_MODELS, f"{base}/front_rgba.png", rgba_bytes, "image/png")
    mask_path = _upload_public(BUCKET_MODELS, f"{base}/front_mask.png", mask_bytes, "image/png")
    svg_path  = _upload_public(BUCKET_MODELS, f"{base}/front_contour.svg", svg_bytes, "image/svg+xml")

    # update row
    supabase.table("recon_jobs").update({
        "front_rgba_path": rgba_path,
        "front_mask_path": mask_path,
        "front_contour_svg_path": svg_path
    }).eq("id", job_id).execute()

    return {"ok": True, "rgba": rgba_path, "mask": mask_path, "svg": svg_path}

@app.post("/fit")
def fit(payload: FitPayload, x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # auth_or_403(x_api_key)

    # Mark running
    supabase.table("recon_jobs").update({"status": "running"}).eq("id", payload.job_id).execute()

    # Fetch job info (paths)
    job = supabase.table("recon_jobs").select("*").eq("id", payload.job_id).single().execute().data
    if not job:
        raise HTTPException(404, "Job not found")

    # OPTIONAL: download photos and run a quick sanity check (omitted for stub)
    # In production: verify they exist, run blur/pose checks, etc.

    try:
        # STUB: create a simple head-like mesh (icosphere) -> GLB
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.09)
        # Lift/elongate a bit to mimic a head
        verts = sphere.vertices.copy()
        verts[:,2] *= 1.3
        verts[:,1] *= 1.1
        sphere.vertices = verts
        glb_bytes = sphere.export(file_type="glb")

        # Upload GLB to public models bucket
        model_path = f"{job['user_id'] or 'anon'}/{payload.job_id}/model.glb"
        up = supabase.storage.from_(BUCKET_MODELS).upload(model_path, glb_bytes, {"contentType":"model/gltf-binary", "upsert": "true"})
        if hasattr(up, 'error') and up.error:
            raise RuntimeError(str(up.error))

        # Update job as done
        supabase.table("recon_jobs").update({
            "status": "done",
            "model_glb_path": model_path,
            "texture_path": None,
            "params_json": {}
        }).eq("id", payload.job_id).execute()

        return {"ok": True}
    except Exception as e:
        supabase.table("recon_jobs").update({
            "status": "error",
            "error_msg": str(e)
        }).eq("id", payload.job_id).execute()
        raise

@app.post("/quality/front")
def quality_front(payload: dict = Body(...), x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # if x_api_key != BACKEND_SECRET:
    #     raise HTTPException(403, "Forbidden")
    job_id = payload.get("job_id")
    landmarks = payload.get("landmarks")  # dict of 6 points {x,y}
    if not job_id or not landmarks:
        raise HTTPException(400, "job_id and landmarks required")

    # read job, get front_path
    job = supabase.table("recon_jobs").select("*").eq("id", job_id).single().execute().data
    if not job: raise HTTPException(404, "job not found")

    # download image
    front_bytes = supabase.storage.from_(BUCKET_RAW).download(job["front_path"])
    if getattr(front_bytes, "error", None):
        raise HTTPException(500, front_bytes.error.message)
    img_np = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img_np.shape[:2]

    # 2D points in image coords, order must match MODEL_POINTS
    def P(k): return (landmarks[k]["x"], landmarks[k]["y"])
    image_points = np.array([
        P("noseTip"), P("chin"),
        P("leftEyeOuter"), P("rightEyeOuter"),
        P("mouthLeft"), P("mouthRight"),
    ], dtype=np.float32)

    # camera intrinsics guess (fx=fy=max(w,h), cx=w/2, cy=h/2)
    f = float(max(w, h))
    camera_matrix = np.array([[f, 0, w/2],
                              [0, f, h/2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise HTTPException(422, "Pose solve failed")

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotationMatrixToEulerAngles(R)

    # blur & exposure
    bv = blur_variance(img_np)
    mean, dark_frac, bright_frac = exposure_metrics(img_np)

    # thresholds
    pose_ok = abs(yaw) <= 10 and abs(pitch) <= 10 and abs(roll) <= 10
    blur_ok = bv >= 120.0
    expo_ok = (80.0 <= mean <= 190.0) and (dark_frac <= 0.35) and (bright_frac <= 0.20)

    passes = pose_ok and blur_ok and expo_ok
    reason = []
    if not pose_ok: reason.append(f"Pose off (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})")
    if not blur_ok: reason.append(f"Too blurry (varLap={bv:.0f} < 120)")
    if not expo_ok: reason.append(f"Lighting issue (mean={mean:.0f}, dark={dark_frac:.2f}, bright={bright_frac:.2f})")
    reason_str = "; ".join(reason) if reason else "OK"

    # persist
    supabase.table("recon_jobs").update({
      "pose_yaw": yaw, "pose_pitch": pitch, "pose_roll": roll,
      "quality_blur": bv,
      "quality_brightness": mean,
      "quality_dark_fraction": dark_frac,
      "quality_bright_fraction": bright_frac,
      "quality_pass": passes,
      "quality_reason": reason_str
    }).eq("id", job_id).execute()

    # Optional: if fail, flip job status to "error" so UI blocks early
    # if not passes:
    #   supabase.table("recon_jobs").update({
    #     "status":"error", "error_msg": reason_str
    #   }).eq("id", job_id).execute()

    return {
      "pass": passes,
      "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
      "blur": bv,
      "exposure": {"mean": mean, "dark_fraction": dark_frac, "bright_fraction": bright_frac},
      "reason": reason_str
    }

@app.post("/quality/left")
def quality_left(payload: dict = Body(...), x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # if x_api_key != BACKEND_SECRET:
    #     raise HTTPException(403, "Forbidden")
    job_id = payload.get("job_id")
    landmarks = payload.get("landmarks")  # dict of 6 points {x,y}
    if not job_id or not landmarks:
        raise HTTPException(400, "job_id and landmarks required")

    # read job, get left_path
    job = supabase.table("recon_jobs").select("*").eq("id", job_id).single().execute().data
    if not job: raise HTTPException(404, "job not found")

    # download image
    left_bytes = supabase.storage.from_(BUCKET_RAW).download(job["left_path"])
    if getattr(left_bytes, "error", None):
        raise HTTPException(500, left_bytes.error.message)
    img_np = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img_np.shape[:2]

    # 2D points in image coords, order must match MODEL_POINTS
    def P(k): return (landmarks[k]["x"], landmarks[k]["y"])
    image_points = np.array([
        P("noseTip"), P("chin"),
        P("leftEyeOuter"), P("rightEyeOuter"),
        P("mouthLeft"), P("mouthRight"),
    ], dtype=np.float32)

    # camera intrinsics guess (fx=fy=max(w,h), cx=w/2, cy=h/2)
    f = float(max(w, h))
    camera_matrix = np.array([[f, 0, w/2],
                              [0, f, h/2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise HTTPException(422, "Pose solve failed")

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotationMatrixToEulerAngles(R)

    # blur & exposure
    bv = blur_variance(img_np)
    mean, dark_frac, bright_frac = exposure_metrics(img_np)

    # left pose thresholds: face should be turned left (yaw ~90°)
    pose_ok = abs(yaw - 90) <= 10 and abs(pitch) <= 10 and abs(roll) <= 10
    blur_ok = bv >= 120.0
    expo_ok = (80.0 <= mean <= 190.0) and (dark_frac <= 0.35) and (bright_frac <= 0.20)

    passes = pose_ok and blur_ok and expo_ok
    reason = []
    if not pose_ok: reason.append(f"Pose off (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})")
    if not blur_ok: reason.append(f"Too blurry (varLap={bv:.0f} < 120)")
    if not expo_ok: reason.append(f"Lighting issue (mean={mean:.0f}, dark={dark_frac:.2f}, bright={bright_frac:.2f})")
    reason_str = "; ".join(reason) if reason else "OK"

    # persist
    supabase.table("recon_jobs").update({
      "left_pose_yaw": yaw, "left_pose_pitch": pitch, "left_pose_roll": roll,
      "left_quality_blur": bv,
      "left_quality_brightness": mean,
      "left_quality_dark_fraction": dark_frac,
      "left_quality_bright_fraction": bright_frac,
      "left_quality_pass": passes,
      "left_quality_reason": reason_str
    }).eq("id", job_id).execute()

    return {
      "pass": passes,
      "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
      "blur": bv,
      "exposure": {"mean": mean, "dark_fraction": dark_frac, "bright_fraction": bright_frac},
      "reason": reason_str
    }

@app.post("/quality/right")
def quality_right(payload: dict = Body(...), x_api_key: str | None = Header(default=None, convert_underscores=False)):
    # if x_api_key != BACKEND_SECRET:
    #     raise HTTPException(403, "Forbidden")
    job_id = payload.get("job_id")
    landmarks = payload.get("landmarks")  # dict of 6 points {x,y}
    if not job_id or not landmarks:
        raise HTTPException(400, "job_id and landmarks required")

    # read job, get right_path
    job = supabase.table("recon_jobs").select("*").eq("id", job_id).single().execute().data
    if not job: raise HTTPException(404, "job not found")

    # download image
    right_bytes = supabase.storage.from_(BUCKET_RAW).download(job["right_path"])
    if getattr(right_bytes, "error", None):
        raise HTTPException(500, right_bytes.error.message)
    img_np = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img_np.shape[:2]

    # 2D points in image coords, order must match MODEL_POINTS
    def P(k): return (landmarks[k]["x"], landmarks[k]["y"])
    image_points = np.array([
        P("noseTip"), P("chin"),
        P("leftEyeOuter"), P("rightEyeOuter"),
        P("mouthLeft"), P("mouthRight"),
    ], dtype=np.float32)

    # camera intrinsics guess (fx=fy=max(w,h), cx=w/2, cy=h/2)
    f = float(max(w, h))
    camera_matrix = np.array([[f, 0, w/2],
                              [0, f, h/2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise HTTPException(422, "Pose solve failed")

    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotationMatrixToEulerAngles(R)

    # blur & exposure
    bv = blur_variance(img_np)
    mean, dark_frac, bright_frac = exposure_metrics(img_np)

    # right pose thresholds: face should be turned right (yaw ~-90°)
    pose_ok = abs(yaw + 90) <= 10 and abs(pitch) <= 10 and abs(roll) <= 10
    blur_ok = bv >= 120.0
    expo_ok = (80.0 <= mean <= 190.0) and (dark_frac <= 0.35) and (bright_frac <= 0.20)

    passes = pose_ok and blur_ok and expo_ok
    reason = []
    if not pose_ok: reason.append(f"Pose off (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})")
    if not blur_ok: reason.append(f"Too blurry (varLap={bv:.0f} < 120)")
    if not expo_ok: reason.append(f"Lighting issue (mean={mean:.0f}, dark={dark_frac:.2f}, bright={bright_frac:.2f})")
    reason_str = "; ".join(reason) if reason else "OK"

    # persist
    supabase.table("recon_jobs").update({
      "right_pose_yaw": yaw, "right_pose_pitch": pitch, "right_pose_roll": roll,
      "right_quality_blur": bv,
      "right_quality_brightness": mean,
      "right_quality_dark_fraction": dark_frac,
      "right_quality_bright_fraction": bright_frac,
      "right_quality_pass": passes,
      "right_quality_reason": reason_str
    }).eq("id", job_id).execute()

    return {
      "pass": passes,
      "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
      "blur": bv,
      "exposure": {"mean": mean, "dark_fraction": dark_frac, "bright_fraction": bright_frac},
      "reason": reason_str
    }

# Vercel handler
handler = app
