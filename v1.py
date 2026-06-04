# """
# =================================================================
# Disaster AI — HuggingFace Spaces API
# Final version — all fixes applied
# =================================================================
# """

# import os
# import io
# import json
# import time
# import base64
# import threading
# import traceback
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import cv2
# import torch
# import requests

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from huggingface_hub import hf_hub_download

# # ════════════════════════════════
# # App Setup
# # ════════════════════════════════
# app = FastAPI(
#     title="Disaster AI Inference API",
#     description="Multi-model disaster scene analysis for Dokai / RoboXavier",
#     version="2.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ════════════════════════════════
# # Configuration
# # ════════════════════════════════
# HF_VICTIM_MODEL_REPO = os.getenv("HF_VICTIM_MODEL_REPO", "")
# ROBOFLOW_API_KEY     = os.getenv("ROBOFLOW_API_KEY", "")
# MODEL_CACHE_DIR      = "/tmp/model_cache"
# os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# TARGET_CLASSES = {
#     0: "injured_civilian",
#     1: "trapped_civilian",
#     2: "safe_civilian",
#     3: "rescue_personnel",
# }

# CLASS_PRIORITY = {
#     "injured_civilian":  1.0,
#     "trapped_civilian":  0.95,
#     "safe_civilian":     0.3,
#     "rescue_personnel":  0.0,
# }

# # ════════════════════════════════
# # Model Registry
# # ════════════════════════════════
# class ModelRegistry:
#     def __init__(self):
#         self._models  = {}
#         self._errors  = {}
#         self._lock    = threading.Lock()

#     def get(self, name):
#         return self._models.get(name)

#     def register(self, name, model):
#         with self._lock:
#             self._models[name] = model
#         print(f"✅ Model registered: {name}")

#     def set_error(self, name, error):
#         with self._lock:
#             self._errors[name] = str(error)
#         print(f"❌ Model error [{name}]: {error}")

#     def is_loaded(self, name):
#         return name in self._models

#     def get_error(self, name):
#         return self._errors.get(name, "Unknown error")

#     def status(self):
#         return {
#             "loaded":  list(self._models.keys()),
#             "errored": {k: v for k, v in self._errors.items()},
#         }

# registry = ModelRegistry()

# # ════════════════════════════════
# # Model Loaders
# # ════════════════════════════════

# def load_ladi_model():
#     """Load LADI-v2 classifier from HuggingFace Hub."""
#     if registry.is_loaded("ladi"):
#         return registry.get("ladi")

#     try:
#         from transformers import AutoImageProcessor, AutoModelForImageClassification

#         print("⬇️  Loading MITLL/LADI-v2-classifier-small ...")

#         processor = AutoImageProcessor.from_pretrained(
#             "MITLL/LADI-v2-classifier-small",
#             cache_dir=MODEL_CACHE_DIR,
#         )

#         model = AutoModelForImageClassification.from_pretrained(
#             "MITLL/LADI-v2-classifier-small",
#             cache_dir=MODEL_CACHE_DIR,
#             trust_remote_code=True,
#             ignore_mismatched_sizes=True,
#         )
#         model.eval()

#         # CPU only on HF free tier
#         registry.register("ladi", {"model": model, "processor": processor})
#         print("✅ LADI-v2 ready")
#         return registry.get("ladi")

#     except Exception as e:
#         print(f"❌ LADI-v2 load failed:\n{traceback.format_exc()}")
#         registry.set_error("ladi", e)
#         return None


# def load_victim_model():
#     """Load YOLOv8 victim detection model from HuggingFace Hub."""
#     if registry.is_loaded("victim"):
#         return registry.get("victim")

#     if not HF_VICTIM_MODEL_REPO:
#         registry.set_error("victim", "HF_VICTIM_MODEL_REPO secret not set — train the model first")
#         return None

#     try:
#         from ultralytics import YOLO

#         print(f"⬇️  Loading victim model from {HF_VICTIM_MODEL_REPO} ...")
#         model_path = hf_hub_download(
#             repo_id=HF_VICTIM_MODEL_REPO,
#             filename="best.pt",
#             cache_dir=MODEL_CACHE_DIR,
#         )
#         model = YOLO(model_path)
#         registry.register("victim", model)
#         print("✅ Victim detection model ready")
#         return model

#     except Exception as e:
#         print(f"❌ Victim model load failed:\n{traceback.format_exc()}")
#         registry.set_error("victim", e)
#         return None


# # ════════════════════════════════
# # Startup — preload everything
# # ════════════════════════════════
# @app.on_event("startup")
# async def startup_event():
#     print("\n" + "="*50)
#     print("🚀 Disaster AI API starting up...")
#     print("="*50)

#     # Always load LADI — it's a public HF model
#     load_ladi_model()

#     # Only load victim model if repo is configured
#     if HF_VICTIM_MODEL_REPO:
#         load_victim_model()
#     else:
#         print("⚠️  Victim model skipped — HF_VICTIM_MODEL_REPO not set")
#         print("   Train the model first, then add the secret to this Space")

#     print("="*50)
#     print(f"📊 Registry status: {registry.status()}")
#     print("="*50 + "\n")


# # ════════════════════════════════
# # Utility
# # ════════════════════════════════
# def read_image(file_bytes: bytes) -> np.ndarray:
#     nparr = np.frombuffer(file_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Invalid image — cannot decode")
#     return img


# def call_roboflow(image: np.ndarray, model_id: str, confidence: int = 40) -> list:
#     if not ROBOFLOW_API_KEY:
#         return []
#     try:
#         _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
#         img_b64 = base64.b64encode(buffer)
#         url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}&confidence={confidence}"
#         res = requests.post(
#             url,
#             data=img_b64,
#             headers={"Content-Type": "application/x-www-form-urlencoded"},
#             timeout=8,
#         )
#         res.raise_for_status()
#         preds = res.json().get("predictions", [])
#         return [
#             {
#                 "class":      p["class"],
#                 "confidence": round(p["confidence"], 4),
#                 "box": {
#                     "xmin": int(p["x"] - p["width"]  / 2),
#                     "ymin": int(p["y"] - p["height"] / 2),
#                     "xmax": int(p["x"] + p["width"]  / 2),
#                     "ymax": int(p["y"] + p["height"] / 2),
#                 },
#             }
#             for p in preds
#         ]
#     except Exception as e:
#         print(f"Roboflow error ({model_id}): {e}")
#         return []


# def compute_triage(detections: list) -> dict:
#     if not detections:
#         return {
#             "total": 0, "critical": 0, "high": 0,
#             "moderate": 0, "low": 0,
#             "highest_score": 0.0,
#             "action": "✅ No victims detected",
#             "ranked_victims": [],
#         }

#     scored = []
#     for d in detections:
#         cls_name = d.get("class", "")
#         conf     = d.get("confidence", 0.5)
#         weight   = CLASS_PRIORITY.get(cls_name, 0.5)
#         score    = round(conf * weight, 4)
#         rank     = (
#             "CRITICAL" if score >= 0.7 else
#             "HIGH"     if score >= 0.4 else
#             "MODERATE" if score >= 0.2 else
#             "LOW"
#         )
#         scored.append({**d, "priority_score": score, "priority_rank": rank})

#     scored.sort(key=lambda x: x["priority_score"], reverse=True)

#     critical = sum(1 for d in scored if d["priority_rank"] == "CRITICAL")
#     high     = sum(1 for d in scored if d["priority_rank"] == "HIGH")
#     moderate = sum(1 for d in scored if d["priority_rank"] == "MODERATE")
#     low      = sum(1 for d in scored if d["priority_rank"] == "LOW")

#     action = (
#         "⚠️ IMMEDIATE RESCUE — Critical victims present"  if critical else
#         "🔴 Deploy rescue team — High priority victims"   if high     else
#         "🟡 Assess and triage — Moderate victims present" if moderate else
#         "🟢 Low priority — Monitor the area"
#     )

#     return {
#         "total":          len(scored),
#         "critical":       critical,
#         "high":           high,
#         "moderate":       moderate,
#         "low":            low,
#         "highest_score":  scored[0]["priority_score"] if scored else 0.0,
#         "action":         action,
#         "ranked_victims": scored,
#     }


# # ════════════════════════════════
# # Routes
# # ════════════════════════════════

# @app.get("/")
# def root():
#     return {
#         "service":   "Disaster AI Inference API",
#         "version":   "2.0.0",
#         "status":    registry.status(),
#         "endpoints": {
#             "GET  /health":          "Health check + model status",
#             "POST /classify":        "LADI-v2 scene classification",
#             "POST /detect/victims":  "Victim detection + triage priority",
#             "POST /detect/vehicles": "Emergency vehicle detection",
#             "POST /analyze/full":    "All models in one call",
#         }
#     }


# @app.get("/health")
# def health():
#     return {
#         "status":        "ok",
#         "registry":      registry.status(),
#         "gpu_available": torch.cuda.is_available(),
#         "timestamp":     time.time(),
#     }


# # ─────────────────────────────────────────────
# # LADI-v2 Classification
# # ─────────────────────────────────────────────
# @app.post("/classify")
# async def classify_scene(
#     file:  UploadFile = File(...),
#     top_k: int = 5,
# ):
#     """
#     Classify disaster scene using LADI-v2.
#     Returns top-k predicted damage categories with confidence scores.
#     """
#     ladi = load_ladi_model()
#     if ladi is None:
#         raise HTTPException(
#             status_code=503,
#             detail=f"LADI-v2 unavailable: {registry.get_error('ladi')}"
#         )

#     contents = await file.read()
#     try:
#         img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image")

#     model     = ladi["model"]
#     processor = ladi["processor"]

#     t0 = time.time()
#     try:
#         inputs = processor(images=img_pil, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

#     elapsed = round((time.time() - t0) * 1000, 2)

#     id2label  = model.config.id2label
#     all_scores = sorted(
#         [
#             {
#                 "class":      id2label[i].lower().replace(" ", "_"),
#                 "confidence": round(float(probs[i]), 4),
#             }
#             for i in range(len(probs))
#         ],
#         key=lambda x: x["confidence"],
#         reverse=True,
#     )

#     # Exclude water/flood from top predictions (not relevant for rover)
#     relevant = [
#         s for s in all_scores
#         if not any(w in s["class"] for w in ["water", "flood"])
#     ]

#     return {
#         "top_predictions":   all_scores[:top_k],
#         "relevant_only":     relevant[:top_k],
#         "all_scores":        all_scores,
#         "inference_time_ms": elapsed,
#     }


# # ─────────────────────────────────────────────
# # Victim Detection
# # ─────────────────────────────────────────────
# @app.post("/detect/victims")
# async def detect_victims(
#     file:       UploadFile = File(...),
#     confidence: float = 0.35,
# ):
#     """
#     Detect victims and classify by triage priority.
#     Returns CRITICAL / HIGH / MODERATE / LOW ranked detections.
#     """
#     model = load_victim_model()
#     if model is None:
#         raise HTTPException(
#             status_code=503,
#             detail=f"Victim model unavailable: {registry.get_error('victim')}"
#         )

#     contents = await file.read()
#     img      = read_image(contents)

#     t0 = time.time()
#     try:
#         results = model.predict(source=img, conf=confidence, verbose=False)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
#     elapsed = round((time.time() - t0) * 1000, 2)

#     raw = []
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             conf_val = float(box.conf[0])
#             cls_id   = int(box.cls[0])
#             raw.append({
#                 "class":      TARGET_CLASSES.get(cls_id, "unknown"),
#                 "class_id":   cls_id,
#                 "confidence": round(conf_val, 4),
#                 "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
#             })

#     triage   = compute_triage(raw)
#     victims  = triage.pop("ranked_victims", raw)

#     return {
#         "detections":        victims,
#         "triage_summary":    triage,
#         "inference_time_ms": elapsed,
#     }


# # ─────────────────────────────────────────────
# # Emergency Vehicle Detection
# # ─────────────────────────────────────────────
# @app.post("/detect/vehicles")
# async def detect_vehicles(file: UploadFile = File(...)):
#     """
#     Detect emergency vehicles using Roboflow.
#     Returns ambulance / fire truck / rescue vehicle detections.
#     """
#     if not ROBOFLOW_API_KEY:
#         raise HTTPException(status_code=503, detail="ROBOFLOW_API_KEY secret not set")

#     contents = await file.read()
#     img = read_image(contents)

#     t0         = time.time()
#     detections = call_roboflow(img, "ambulance-4bova/1", confidence=40)
#     elapsed    = round((time.time() - t0) * 1000, 2)

#     has_ambulance  = any("ambulance"  in d["class"].lower() for d in detections)
#     has_fire_truck = any("fire"       in d["class"].lower() for d in detections)

#     return {
#         "detections": detections,
#         "emergency_vehicles": {
#             "ambulance_detected":  has_ambulance,
#             "fire_truck_detected": has_fire_truck,
#             "rescue_arrived":      has_ambulance or has_fire_truck,
#         },
#         "inference_time_ms": elapsed,
#     }


# # ─────────────────────────────────────────────
# # Full Analysis — all models in one call
# # ─────────────────────────────────────────────
# @app.post("/analyze/full")
# async def full_analysis(
#     file:         UploadFile = File(...),
#     run_victims:  bool = True,
#     run_vehicles: bool = True,
#     run_classify: bool = True,
# ):
#     """
#     Run all available models on one image.
#     This is the main endpoint your rover Flask app should call.

#     Returns unified JSON with zone color, all detections, triage summary.
#     """
#     contents = await file.read()
#     t_total  = time.time()
#     output   = {}

#     # ── LADI classification ──
#     if run_classify:
#         try:
#             fake_file = UploadFile(filename="f.jpg", file=io.BytesIO(contents))
#             output["classification"] = await classify_scene(fake_file)
#         except HTTPException as e:
#             output["classification"] = {"error": e.detail}
#         except Exception as e:
#             output["classification"] = {"error": str(e)}

#     # ── Victim detection ──
#     if run_victims:
#         try:
#             fake_file = UploadFile(filename="f.jpg", file=io.BytesIO(contents))
#             output["victims"] = await detect_victims(fake_file)
#         except HTTPException as e:
#             output["victims"] = {"error": e.detail}
#         except Exception as e:
#             output["victims"] = {"error": str(e)}

#     # ── Vehicle detection ──
#     if run_vehicles:
#         try:
#             fake_file = UploadFile(filename="f.jpg", file=io.BytesIO(contents))
#             output["vehicles"] = await detect_vehicles(fake_file)
#         except HTTPException as e:
#             output["vehicles"] = {"error": e.detail}
#         except Exception as e:
#             output["vehicles"] = {"error": str(e)}

#     # ── Zone color ──
#     triage_data  = output.get("victims",        {}).get("triage_summary", {})
#     classify_top = output.get("classification", {}).get("top_predictions", [{}])
#     top_class    = classify_top[0].get("class", "") if classify_top else ""

#     critical = triage_data.get("critical", 0)
#     high     = triage_data.get("high",     0)

#     if critical > 0 or any(w in top_class for w in ["destroy", "collapse", "major"]):
#         zone_color = "red"
#     elif high > 0 or "minor_damage" in top_class:
#         zone_color = "orange"
#     elif triage_data.get("total", 0) > 0:
#         zone_color = "yellow"
#     else:
#         zone_color = "green"

#     return {
#         "zone_color":    zone_color,
#         "results":       output,
#         "total_time_ms": round((time.time() - t_total) * 1000, 2),
#         "timestamp":     time.time(),
#     }


# # ════════════════════════════════
# # Entry Point
# # ════════════════════════════════
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)


"""
=================================================================
Disaster AI — Local API Client
=================================================================
Run this on your laptop to call your HuggingFace Spaces API.
No AI libraries needed locally — just requests + opencv.

Usage:
    python disaster_ai_client.py --image path/to/image.jpg
    python disaster_ai_client.py --webcam
    python disaster_ai_client.py --video path/to/video.mp4
=================================================================
"""

import os
import cv2
import sys
import time
import json
import argparse
import requests
import threading
from datetime import datetime

# ════════════════════════════════
# Configuration — edit these
# ════════════════════════════════
HF_API_URL = "https://egoisticcoderx-dokai-inference-api.hf.space"

# Colab fallback URLs (paste from ngrok when running)
COLAB_URLS = {
    "model1": os.getenv("COLAB_MODEL_1_URL", ""),  # xView2
    "model2": os.getenv("COLAB_MODEL_2_URL", ""),  # Fire & Smoke
    "model3": os.getenv("COLAB_MODEL_3_URL", ""),  # LADI
    "model7": os.getenv("COLAB_MODEL_7_URL", ""),  # Victim Detection
}

# ════════════════════════════════
# Core API Client Class
# ════════════════════════════════
class DisasterAIClient:
    def __init__(self, base_url=HF_API_URL, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = requests.Session()
        self._alive_thread = None

        print(f"🔗 Connected to: {self.base_url}")

    # ─────────────────────────────
    # Health check
    # ─────────────────────────────
    def health(self):
        """Check if API is online and which models are loaded."""
        try:
            res = self.session.get(f"{self.base_url}/health", timeout=10)
            res.raise_for_status()
            data = res.json()
            print("\n📡 API Health:")
            print(f"   Status:  {data.get('status')}")
            print(f"   Loaded:  {data.get('registry', {}).get('loaded', [])}")
            print(f"   Errors:  {data.get('registry', {}).get('errored', {})}")
            print(f"   GPU:     {data.get('gpu_available')}")
            return data
        except Exception as e:
            print(f"❌ API unreachable: {e}")
            return None

    # ─────────────────────────────
    # Send image bytes to endpoint
    # ─────────────────────────────
    def _post_image(self, endpoint, image_bytes, params=None):
        """
        Internal — POST image bytes to an endpoint.
        Returns parsed JSON or None on failure.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            res = self.session.post(
                url,
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                params=params or {},
                timeout=self.timeout,
            )
            if res.status_code == 503:
                print(f"⚠️  {endpoint}: model unavailable — {res.json().get('detail', '')}")
                return None
            res.raise_for_status()
            return res.json()
        except requests.exceptions.Timeout:
            print(f"⏱️  {endpoint}: request timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"❌ {endpoint} error: {e}")
            return None

    def _encode_frame(self, frame):
        """Convert OpenCV frame to JPEG bytes."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    # ─────────────────────────────
    # Individual endpoints
    # ─────────────────────────────
    def classify(self, frame, top_k=5):
        """LADI-v2 scene classification."""
        return self._post_image("classify", self._encode_frame(frame), {"top_k": top_k})

    def detect_victims(self, frame, confidence=0.35):
        """Victim detection + triage priority."""
        return self._post_image("detect/victims", self._encode_frame(frame), {"confidence": confidence})

    def detect_vehicles(self, frame):
        """Emergency vehicle detection."""
        return self._post_image("detect/vehicles", self._encode_frame(frame))

    def analyze_full(self, frame, run_victims=True, run_vehicles=True, run_classify=True):
        """
        All models in one call — use this for the rover.
        Returns zone_color + all detections in one response.
        """
        return self._post_image(
            "analyze/full",
            self._encode_frame(frame),
            {
                "run_victims":  run_victims,
                "run_vehicles": run_vehicles,
                "run_classify": run_classify,
            }
        )

    # ─────────────────────────────
    # Keep-alive ping
    # ─────────────────────────────
    def start_keepalive(self, interval=600):
        """
        Ping the API every `interval` seconds to prevent HF cold start.
        Call this once after creating the client.
        """
        def _ping():
            while True:
                try:
                    self.session.get(f"{self.base_url}/health", timeout=5)
                    print(f"[{datetime.now().strftime('%H:%M')}] 💓 API keep-alive ping sent")
                except Exception:
                    pass
                time.sleep(interval)

        self._alive_thread = threading.Thread(target=_ping, daemon=True)
        self._alive_thread.start()
        print(f"💓 Keep-alive started (every {interval//60} min)")


# ════════════════════════════════
# Result Printer
# ════════════════════════════════
def print_result(data, mode="full"):
    """Pretty print API response to terminal."""
    if data is None:
        print("❌ No result")
        return

    print("\n" + "═"*50)

    if mode == "full":
        zone = data.get("zone_color", "unknown").upper()
        colors = {"RED": "🔴", "ORANGE": "🟠", "YELLOW": "🟡", "GREEN": "🟢"}
        print(f"{colors.get(zone, '⚪')} ZONE: {zone}")
        print(f"⏱  Total time: {data.get('total_time_ms')}ms")

        results = data.get("results", {})

        # Victims
        victims = results.get("victims", {})
        if "error" not in victims:
            triage = victims.get("triage_summary", {})
            print(f"\n👥 VICTIMS:")
            print(f"   Total:    {triage.get('total', 0)}")
            print(f"   Critical: {triage.get('critical', 0)}")
            print(f"   High:     {triage.get('high', 0)}")
            print(f"   Action:   {triage.get('action', '-')}")
        else:
            print(f"\n👥 VICTIMS: {victims['error']}")

        # Classification
        classify = results.get("classification", {})
        if "error" not in classify:
            top = classify.get("top_predictions", [])[:3]
            print(f"\n🏚  SCENE:")
            for t in top:
                bar = "█" * int(t["confidence"] * 20)
                print(f"   {t['class']:<30} {bar} {t['confidence']:.2f}")
        else:
            print(f"\n🏚  SCENE: {classify['error']}")

        # Vehicles
        vehicles = results.get("vehicles", {})
        if "error" not in vehicles:
            ev = vehicles.get("emergency_vehicles", {})
            print(f"\n🚑 VEHICLES:")
            print(f"   Ambulance:     {'✅' if ev.get('ambulance_detected')  else '❌'}")
            print(f"   Fire truck:    {'✅' if ev.get('fire_truck_detected') else '❌'}")
            print(f"   Rescue arrived: {'✅' if ev.get('rescue_arrived')      else '❌'}")
        else:
            print(f"\n🚑 VEHICLES: {vehicles['error']}")

    elif mode == "classify":
        top = data.get("top_predictions", [])[:5]
        print("🏚  SCENE CLASSIFICATION:")
        for t in top:
            bar = "█" * int(t["confidence"] * 20)
            print(f"  {t['class']:<30} {bar} {t['confidence']:.2f}")

    elif mode == "victims":
        triage = data.get("triage_summary", {})
        print(f"👥 TRIAGE: {triage.get('action', '-')}")
        for d in data.get("detections", []):
            print(f"  [{d['priority_rank']:8}] {d['class']:<20} conf={d['confidence']:.2f}  score={d['priority_score']:.2f}")

    print("═"*50 + "\n")


# ════════════════════════════════
# Draw boxes on frame
# ════════════════════════════════
COLORS = {
    "injured_civilian":  (0,   0,   255),  # red
    "trapped_civilian":  (0,   165, 255),  # orange
    "safe_civilian":     (0,   255, 0),    # green
    "rescue_personnel":  (255, 255, 0),    # yellow
    "ambulance":         (0,   255, 255),  # cyan
}
RANK_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH":     (0, 165, 255),
    "MODERATE": (0, 255, 255),
    "LOW":      (0, 255, 0),
}

def draw_results(frame, api_response):
    """Draw bounding boxes and zone color on frame."""
    if api_response is None:
        return frame

    annotated = frame.copy()
    results   = api_response.get("results", {})

    # Draw victim boxes
    for d in results.get("victims", {}).get("detections", []):
        box   = d.get("box", {})
        if not box:
            continue
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        color = RANK_COLORS.get(d.get("priority_rank", "LOW"), (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} [{d.get('priority_rank','?')}] {d['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw vehicle boxes
    for d in results.get("vehicles", {}).get("detections", []):
        box = d.get("box", {})
        if not box:
            continue
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        color = COLORS.get(d["class"].lower(), (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{d['class']} {d['confidence']:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Zone color banner at top
    zone  = api_response.get("zone_color", "green")
    bcolors = {"red": (0,0,255), "orange": (0,165,255), "yellow": (0,255,255), "green": (0,200,0)}
    bcolor  = bcolors.get(zone, (200, 200, 200))
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), bcolor, -1)
    cv2.putText(annotated, f"ZONE: {zone.upper()}  |  {datetime.now().strftime('%H:%M:%S')}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Triage action text
    action = (results.get("victims", {})
                     .get("triage_summary", {})
                     .get("action", ""))
    if action:
        cv2.putText(annotated, action, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


# ════════════════════════════════
# Modes
# ════════════════════════════════
def run_image(client, path):
    """Analyze a single image file."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"❌ Cannot read image: {path}")
        return

    print(f"📸 Analyzing: {path}")
    result = client.analyze_full(frame)
    print_result(result)

    # Save annotated image
    annotated = draw_results(frame, result)
    out_path  = f"result_{os.path.basename(path)}"
    cv2.imwrite(out_path, annotated)
    print(f"💾 Saved annotated image: {out_path}")


def run_video(client, path, show=True):
    """Analyze a video file frame by frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {path}")
        return

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    skip      = max(int(fps), 1)   # process 1 frame per second
    count     = 0
    last_result = None

    print(f"🎬 Processing video: {path} @ {fps:.0f}fps (1 API call/sec)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            print(f"\n⏱  t={count/fps:.1f}s — sending frame to API...")
            last_result = client.analyze_full(frame)
            print_result(last_result)

        if show and last_result:
            annotated = draw_results(frame, last_result)
            cv2.imshow("Disaster AI", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1

    cap.release()
    cv2.destroyAllWindows()


def run_webcam(client, cam_index=0):
    """
    Live webcam feed — sends frame every 3 seconds to API,
    displays latest result continuously.
    Optimized for your 20m range use case.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Cannot open webcam {cam_index}")
        return

    print(f"📷 Webcam live feed started (press Q to quit)")

    last_result     = None
    last_send_time  = 0
    send_interval   = 3.0    # seconds between API calls
    pending         = False

    def send_frame(frame):
        """Called in background thread — doesn't block display."""
        nonlocal last_result, pending
        pending     = True
        last_result = client.analyze_full(frame)
        print_result(last_result)
        pending     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # Send frame to API every N seconds
        if now - last_send_time >= send_interval and not pending:
            last_send_time = now
            t = threading.Thread(
                target=send_frame,
                args=(frame.copy(),),
                daemon=True,
            )
            t.start()

        # Always show annotated frame with latest result
        display = draw_results(frame, last_result) if last_result else frame

        # Show pending indicator
        if pending:
            cv2.putText(display, "⏳ Analyzing...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Disaster AI — Live", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📷 Webcam stopped")


# ════════════════════════════════
# CLI Entry Point
# ════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster AI Local Client")
    parser.add_argument("--url",    default=HF_API_URL, help="API base URL")
    parser.add_argument("--image",  help="Path to image file")
    parser.add_argument("--video",  help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam")
    parser.add_argument("--health", action="store_true", help="Check API health only")
    parser.add_argument("--cam",    type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args()

    client = DisasterAIClient(base_url=args.url)
    client.start_keepalive(interval=600)

    if args.health or (not args.image and not args.video and not args.webcam):
        client.health()
        sys.exit(0)

    if args.image:
        run_image(client, args.image)

    elif args.video:
        run_video(client, args.video)

    elif args.webcam:
        run_webcam(client, cam_index=args.cam)
