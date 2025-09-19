import os, io, uuid, tempfile
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import trimesh
import numpy as np

load_dotenv()
app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BACKEND_SECRET = os.environ["BACKEND_SECRET"]
BUCKET_RAW = os.environ.get("BUCKET_RAW", "raw-photos")
BUCKET_MODELS = os.environ.get("BUCKET_MODELS", "models")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class FitPayload(BaseModel):
    job_id: str

# def auth_or_403(x_api_key: str | None):
#     if x_api_key != BACKEND_SECRET:
#         raise HTTPException(status_code=403, detail="Forbidden")

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
