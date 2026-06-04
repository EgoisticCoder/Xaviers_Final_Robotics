"""
=================================================================
Disaster AI Client — v3.0
=================================================================
New in v3:
  - Webcam preview with OpenCV-annotated frame returned to browser
  - 4 Roboflow placeholder models running in parallel threads
  - Local YOLO fire & smoke detection
  - Local YOLO posture model (companion to victim detection)
  - Image preprocessing for accuracy (CLAHE + sharpening)
  - Enhanced Groq prompt with chain-of-thought verification

Run:
    pip install flask flask-cors requests opencv-python numpy ultralytics
    python v2.py --serve --groq YOUR_GROQ_KEY
=================================================================
"""

import os, io, cv2, sys, json, math, time, uuid, base64
import heapq, argparse, threading, traceback
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# ════════════════════════════════
# Configuration
# ════════════════════════════════
HF_API_URL   = os.getenv("HF_API_URL",   "https://egoisticcoderx-dokai-inference-api.hf.space")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

# ── Local YOLO model paths (set via env or defaults) ──
FIRE_SMOKE_MODEL_PATH = os.getenv("FIRE_SMOKE_MODEL_PATH", "models/fire_smoke.pt")
POSTURE_MODEL_PATH    = os.getenv("POSTURE_MODEL_PATH",    "models/posture.pt")
VICTIM_MODEL_PATH     = os.getenv("VICTIM_MODEL_PATH",     "models/victim.pt")

# ── Roboflow placeholder models (change model_id when ready) ──
# Set enabled=False for models you haven't trained yet —
# they will be silently skipped without breaking anything.
ROBOFLOW_MODELS = {
    "rf_custom_1": {
        "model_id":  "PLACEHOLDER_MODEL_1/1",   # ← change this
        "name":      "Custom Model 1",
        "enabled":   False,                      # ← set True when ready
        "confidence": 40,
    },
    "rf_custom_2": {
        "model_id":  "PLACEHOLDER_MODEL_2/1",   # ← change this
        "name":      "Custom Model 2",
        "enabled":   False,
        "confidence": 40,
    },
    "rf_custom_3": {
        "model_id":  "PLACEHOLDER_MODEL_3/1",   # ← change this
        "name":      "Custom Model 3",
        "enabled":   False,
        "confidence": 40,
    },
    "rf_custom_4": {
        "model_id":  "PLACEHOLDER_MODEL_4/1",   # ← change this
        "name":      "Custom Model 4",
        "enabled":   False,
        "confidence": 40,
    },
}

TARGET_CLASSES = {0: "injured_civilian", 1: "trapped_civilian",
                  2: "safe_civilian",    3: "rescue_personnel"}
CLASS_PRIORITY = {"injured_civilian": 1.0, "trapped_civilian": 0.95,
                  "safe_civilian": 0.3,    "rescue_personnel": 0.0}

POSTURE_LABELS = {
    0: "standing",  1: "sitting",  2: "lying_down",
    3: "crawling",  4: "collapsed", 5: "waving",
}
POSTURE_RISK = {
    "standing": 0.2, "sitting": 0.4, "lying_down": 0.7,
    "crawling": 0.6, "collapsed": 1.0, "waving": 0.5,
}


# ════════════════════════════════════════════════════════
# 1. IMAGE PREPROCESSING (accuracy boost)
# ════════════════════════════════════════════════════════
def preprocess_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE + mild sharpening to improve detection accuracy,
    especially in low-light or dusty disaster scenes.
    """
    # Convert to LAB, apply CLAHE to L channel
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp mask for edge sharpening
    blur      = cv2.GaussianBlur(enhanced, (0, 0), 2.5)
    sharpened = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

    return sharpened


def encode_frame(frame: np.ndarray, quality: int = 85, max_dim: int = 640) -> bytes:
    """Resize (keep aspect) + JPEG encode."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ════════════════════════════════════════════════════════
# 2. LOCAL YOLO MODELS
# ════════════════════════════════════════════════════════
class LocalYOLOModels:
    """
    Manages three optional local YOLO models:
      - Victim detection (custom trained)
      - Fire & smoke detection
      - Posture estimation (companion to victim model)

    All models load lazily and fail gracefully — if the .pt file
    doesn't exist, the model is skipped silently.
    """

    def __init__(self):
        self._models = {}
        self._lock   = threading.Lock()

    def _load(self, name: str, path: str, labels: dict = None):
        if name in self._models:
            return self._models[name]
        if not os.path.exists(path):
            print(f"⚠️  [{name}] model not found at {path} — skipped")
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            with self._lock:
                self._models[name] = {"model": model, "labels": labels}
            print(f"✅ [{name}] loaded from {path}")
            return self._models[name]
        except Exception as e:
            print(f"❌ [{name}] load failed: {e}")
            return None

    def load_all(self):
        """Call once at startup to preload everything."""
        self._load("victim",     VICTIM_MODEL_PATH,     TARGET_CLASSES)
        self._load("fire_smoke", FIRE_SMOKE_MODEL_PATH, {0:"fire", 1:"smoke", 2:"flame"})
        self._load("posture",    POSTURE_MODEL_PATH,    POSTURE_LABELS)

    def run_victim(self, frame: np.ndarray, conf: float = 0.30) -> dict:
        entry = self._models.get("victim")
        if not entry:
            return {"detections": [], "triage_summary": {}, "source": "unavailable"}
        try:
            results = entry["model"](frame, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = TARGET_CLASSES.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                weight = CLASS_PRIORITY.get(cls_name, 0.5)
                score  = round(conf_val * weight, 4)
                rank   = ("CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.4
                          else "MODERATE" if score >= 0.2 else "LOW")
                dets.append({
                    "class": cls_name, "class_id": cls_id,
                    "confidence": round(conf_val, 4),
                    "priority_score": score, "priority_rank": rank,
                    "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                    "source": "local_victim_yolo",
                })
            dets.sort(key=lambda d: d["priority_score"], reverse=True)
            critical = sum(1 for d in dets if d["priority_rank"] == "CRITICAL")
            high     = sum(1 for d in dets if d["priority_rank"] == "HIGH")
            action   = ("⚠️ IMMEDIATE RESCUE" if critical else
                       "🔴 Deploy rescue team" if high else
                       "🟡 Triage required" if dets else "✅ Clear")
            return {
                "detections": dets,
                "triage_summary": {
                    "total": len(dets), "critical": critical, "high": high,
                    "action": action,
                },
                "source": "local_victim_yolo",
            }
        except Exception as e:
            return {"detections": [], "error": str(e), "source": "local_victim_yolo"}

    def run_fire_smoke(self, frame: np.ndarray, conf: float = 0.35) -> list:
        entry = self._models.get("fire_smoke")
        if not entry:
            return []
        try:
            results = entry["model"](frame, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                labels   = entry["labels"] or {}
                cls_name = labels.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append({
                    "class": cls_name, "confidence": round(conf_val, 4),
                    "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                    "source": "local_fire_smoke_yolo",
                })
            return dets
        except Exception as e:
            print(f"fire_smoke inference error: {e}")
            return []

    def run_posture(self, frame: np.ndarray, conf: float = 0.30) -> list:
        """
        Companion to victim detection — estimates body posture.
        Returns posture classifications with risk scores.
        Higher risk = person likely needs help.
        """
        entry = self._models.get("posture")
        if not entry:
            return []
        try:
            results = entry["model"](frame, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                labels   = entry["labels"] or {}
                posture  = labels.get(cls_id, f"posture_{cls_id}")
                risk     = POSTURE_RISK.get(posture, 0.5)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append({
                    "class": posture, "confidence": round(conf_val, 4),
                    "risk_score": round(risk, 3),
                    "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                    "source": "local_posture_yolo",
                })
            dets.sort(key=lambda d: d["risk_score"], reverse=True)
            return dets
        except Exception as e:
            print(f"posture inference error: {e}")
            return []

    def status(self) -> dict:
        return {name: "loaded" for name in self._models}


yolo_models = LocalYOLOModels()


# ════════════════════════════════════════════════════════
# 3. ROBOFLOW PARALLEL RUNNER
# ════════════════════════════════════════════════════════
class RoboflowRunner:
    """
    Runs multiple Roboflow models in parallel threads.
    Placeholder models that don't exist yet are silently skipped.
    When you're ready to use a model, set enabled=True and
    replace the model_id in ROBOFLOW_MODELS at the top of the file.
    """

    def __init__(self, api_key: str, model_configs: dict):
        self.api_key = api_key
        self.configs = model_configs
        self.enabled = bool(api_key)
        if not self.enabled:
            print("⚠️  ROBOFLOW_API_KEY not set — Roboflow models disabled")

    def _call_single(self, key: str, cfg: dict, image: np.ndarray) -> dict:
        """
        Call one Roboflow model. Returns {"key": ..., "detections": [...]}
        Never raises — all errors return empty detections.
        """
        result = {"key": key, "name": cfg["name"], "detections": []}
        if not cfg.get("enabled", False):
            return result  # placeholder — silently skip

        model_id = cfg["model_id"]
        # Skip if still placeholder
        if "PLACEHOLDER" in model_id.upper():
            return result

        try:
            max_dim = 640
            h, w    = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            _, buf  = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64     = base64.b64encode(buf)
            url     = (f"https://detect.roboflow.com/{model_id}"
                       f"?api_key={self.api_key}&confidence={cfg.get('confidence', 40)}")
            res = requests.post(
                url, data=b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=8,
            )
            if res.status_code in (404, 401, 403):
                # Model doesn't exist yet or wrong key — skip silently
                return result
            res.raise_for_status()
            preds = res.json().get("predictions", [])
            result["detections"] = [
                {
                    "class":      p["class"],
                    "confidence": round(p["confidence"], 4),
                    "source":     f"roboflow_{key}",
                    "box": {
                        "xmin": int(p["x"] - p["width"]  / 2),
                        "ymin": int(p["y"] - p["height"] / 2),
                        "xmax": int(p["x"] + p["width"]  / 2),
                        "ymax": int(p["y"] + p["height"] / 2),
                    },
                }
                for p in preds
            ]
        except requests.Timeout:
            pass  # silently skip on timeout
        except Exception:
            pass  # silently skip on any error

        return result

    def run_all(self, image: np.ndarray) -> dict:
        """
        Run all enabled Roboflow models in parallel.
        Returns merged dict: {"rf_custom_1": [...], "rf_custom_2": [...], ...}
        Always returns within ~8s (capped by timeout per model).
        """
        if not self.enabled:
            return {}

        all_results = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(self._call_single, key, cfg, image.copy()): key
                for key, cfg in self.configs.items()
            }
            for fut in as_completed(futures, timeout=10):
                try:
                    res = fut.result()
                    all_results[res["key"]] = {
                        "name":       res["name"],
                        "detections": res["detections"],
                    }
                except Exception:
                    pass

        return all_results


rf_runner = RoboflowRunner(ROBOFLOW_API_KEY, ROBOFLOW_MODELS)


# ════════════════════════════════════════════════════════
# 4. GROQ VERIFICATION LAYER (enhanced accuracy)
# ════════════════════════════════════════════════════════
class GroqVerifier:
    MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.enabled = bool(api_key)
        if not self.enabled:
            print("⚠️  GROQ_API_KEY not set — verification disabled")

    def verify(self, frame: np.ndarray, detections: list,
               classifications: list, fire_smoke: list = None,
               posture: list = None) -> dict:
        """
        Chain-of-thought verification:
          Step 1 — Scout examines the scene holistically
          Step 2 — Verifies each detection individually
          Step 3 — Returns confirmed / overridden result + accuracy score
        """
        if not self.enabled:
            return self._passthrough(detections, classifications)

        # Preprocess + encode at higher quality for Groq
        processed = preprocess_for_detection(frame)
        h, w = processed.shape[:2]
        scale = min(1.0, 800 / max(h, w))
        sized = cv2.resize(processed, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", sized, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf).decode()

        context = {
            "victim_detections":    detections,
            "scene_classifications": classifications,
            "fire_smoke":           fire_smoke or [],
            "posture_detections":   posture or [],
        }

        prompt = f"""You are an expert disaster response AI analyst.

TASK: Verify the following AI detections on the disaster scene image.

DETECTIONS TO VERIFY:
{json.dumps(context, indent=2)}

INSTRUCTIONS (follow in order):
1. EXAMINE the image — note visible people, damage, fire/smoke, debris
2. For EACH victim detection: confirm class or correct it
   (classes: injured_civilian, trapped_civilian, safe_civilian, rescue_personnel)
3. For EACH fire/smoke detection: confirm or reject
4. Rate overall detection ACCURACY 1-10 based on what you actually see

RESPOND ONLY with this exact JSON (no other text, no markdown):
{{
  "scene_observation": "1-2 sentences describing what you actually see",
  "verified_detections": [
    {{
      "original_class": "...",
      "verified_class": "...",
      "action": "CONFIRMED",
      "confidence": 0.0,
      "reason": ""
    }}
  ],
  "verified_classifications": [
    {{
      "original_class": "...",
      "verified_class": "...",
      "action": "CONFIRMED",
      "confidence": 0.0
    }}
  ],
  "fire_smoke_confirmed": true,
  "accuracy_score": 8,
  "critical_findings": "any critical observations not caught by models"
}}"""

        try:
            res = requests.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": self.MODEL, "max_tokens": 1000, "temperature": 0.05,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }],
                },
                timeout=28,
            )
            res.raise_for_status()
            raw = res.json()["choices"][0]["message"]["content"].strip()
            # Strip fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())

            # Apply overrides
            v_dets    = parsed.get("verified_detections", [])
            v_cls     = parsed.get("verified_classifications", [])
            overrides = 0

            final_dets = []
            for i, det in enumerate(detections):
                if i < len(v_dets):
                    v = v_dets[i]
                    if v.get("action") == "OVERRIDDEN":
                        overrides += 1
                        det = {**det,
                               "class":        v.get("verified_class", det.get("class")),
                               "groq_verified": True, "groq_action": "OVERRIDDEN",
                               "groq_reason":   v.get("reason", "")}
                    else:
                        det = {**det, "groq_verified": True, "groq_action": "CONFIRMED"}
                final_dets.append(det)

            final_cls = []
            for i, cls in enumerate(classifications):
                if i < len(v_cls):
                    v = v_cls[i]
                    if v.get("action") == "OVERRIDDEN":
                        overrides += 1
                        cls = {**cls,
                               "class":         v.get("verified_class", cls.get("class")),
                               "groq_verified":  True, "groq_action": "OVERRIDDEN"}
                    else:
                        cls = {**cls, "groq_verified": True, "groq_action": "CONFIRMED"}
                final_cls.append(cls)

            return {
                "verified_detections":      final_dets,
                "verified_classifications": final_cls,
                "accuracy_score":           parsed.get("accuracy_score"),
                "groq_summary":             parsed.get("scene_observation", ""),
                "critical_findings":        parsed.get("critical_findings", ""),
                "fire_smoke_confirmed":     parsed.get("fire_smoke_confirmed", False),
                "overrides_applied":        overrides,
            }

        except json.JSONDecodeError as e:
            print(f"⚠️  Groq parse error: {e}")
            return self._passthrough(detections, classifications)
        except Exception as e:
            print(f"⚠️  Groq error: {e}")
            return self._passthrough(detections, classifications)

    def _passthrough(self, detections, classifications):
        return {
            "verified_detections":      detections,
            "verified_classifications": classifications,
            "accuracy_score":           None,
            "groq_summary":             "Verification skipped",
            "critical_findings":        "",
            "fire_smoke_confirmed":     False,
            "overrides_applied":        0,
        }


# ════════════════════════════════════════════════════════
# 5. A* GPS PATH PLANNER
# ════════════════════════════════════════════════════════
class PathPlanner:
    def __init__(self, grid_resolution: float = 1.0):
        self.resolution = grid_resolution

    def gps_to_meters(self, lat, lng, origin_lat, origin_lng):
        x = (lng - origin_lng) * 111320 * math.cos(math.radians(origin_lat))
        y = (lat - origin_lat) * 111320
        return x, y

    def meters_to_gps(self, x, y, origin_lat, origin_lng):
        lat = origin_lat + y / 111320
        lng = origin_lng + x / (111320 * math.cos(math.radians(origin_lat)))
        return lat, lng

    def meters_to_cell(self, x, y):
        return int(round(x / self.resolution)), int(round(y / self.resolution))

    def cell_to_meters(self, cx, cy):
        return cx * self.resolution, cy * self.resolution

    def _astar(self, start_cell, goal_cell, blocked: set):
        def h(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
        heap    = [(h(start_cell, goal_cell), 0, start_cell, [start_cell])]
        visited = {}
        while heap:
            f, g, cur, path = heapq.heappop(heap)
            if cur in visited: continue
            visited[cur] = g
            if cur == goal_cell: return path
            cx, cy = cur
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nb = (cx+dx, cy+dy)
                if nb in blocked or nb in visited: continue
                ng = g + math.hypot(dx, dy) * self.resolution
                heapq.heappush(heap, (ng + h(nb, goal_cell), ng, nb, path + [nb]))
        return None

    def plan(self, start_gps: dict, targets_gps: list,
             obstacles_gps: list, obstacle_radius: float = 2.0) -> dict:
        if not targets_gps:
            return {"path": [], "visit_order": [], "total_distance": 0, "blocked_targets": []}

        origin_lat = start_gps["lat"]
        origin_lng = start_gps["lng"]

        blocked = set()
        for obs in obstacles_gps:
            ox, oy   = self.gps_to_meters(obs["lat"], obs["lng"], origin_lat, origin_lng)
            r        = obs.get("radius", obstacle_radius)
            cells_r  = int(math.ceil(r / self.resolution)) + 1
            cx0, cy0 = self.meters_to_cell(ox, oy)
            for dcx in range(-cells_r, cells_r+1):
                for dcy in range(-cells_r, cells_r+1):
                    if math.hypot(dcx, dcy) * self.resolution <= r:
                        blocked.add((cx0+dcx, cy0+dcy))

        sx, sy  = self.gps_to_meters(start_gps["lat"], start_gps["lng"], origin_lat, origin_lng)
        start_c = self.meters_to_cell(sx, sy)

        target_cells = []
        for t in targets_gps:
            tx, ty = self.gps_to_meters(t["lat"], t["lng"], origin_lat, origin_lng)
            target_cells.append((self.meters_to_cell(tx, ty), t.get("id", str(uuid.uuid4())[:6])))

        # Nearest-neighbour TSP
        remaining    = list(range(len(target_cells)))
        visit_order  = []
        current_cell = start_c
        while remaining:
            best_idx = min(remaining,
                           key=lambda i: math.hypot(
                               target_cells[i][0][0]-current_cell[0],
                               target_cells[i][0][1]-current_cell[1]))
            visit_order.append(best_idx)
            current_cell = target_cells[best_idx][0]
            remaining.remove(best_idx)

        full_path       = []
        total_dist      = 0.0
        blocked_targets = []
        prev_cell       = start_c

        for idx in visit_order:
            goal_cell, tid = target_cells[idx]
            segment = self._astar(prev_cell, goal_cell, blocked)
            if segment is None:
                blocked_targets.append(tid)
                continue
            for cell in segment[1:]:
                mx, my = self.cell_to_meters(*cell)
                glat, glng = self.meters_to_gps(mx, my, origin_lat, origin_lng)
                full_path.append({"lat": glat, "lng": glng})
                if len(full_path) >= 2:
                    p = full_path[-2]
                    total_dist += math.hypot(glat-p["lat"], glng-p["lng"]) * 111320
            prev_cell = goal_cell

        return {
            "path":            full_path,
            "visit_order":     [target_cells[i][1] for i in visit_order],
            "total_distance":  round(total_dist, 2),
            "blocked_targets": blocked_targets,
        }


# ════════════════════════════════════════════════════════
# 6. DISASTER AI CLIENT
# ════════════════════════════════════════════════════════
class DisasterAIClient:
    def __init__(self, base_url=HF_API_URL, groq_key=GROQ_API_KEY, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = requests.Session()
        self.groq     = GroqVerifier(groq_key)
        print(f"🔗 HF API: {self.base_url}")
        print(f"🧠 Groq:   {'enabled' if self.groq.enabled else 'disabled'}")

    def _post(self, endpoint, img_bytes, params=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            res = self.session.post(
                url, files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
                params=params or {}, timeout=self.timeout)
            if res.status_code == 503:
                return {"error": res.json().get("detail", "503")}
            res.raise_for_status()
            return res.json()
        except requests.Timeout:
            return {"error": f"timeout after {self.timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    def analyze(self, frame: np.ndarray, verify=True) -> dict:
        """
        Full pipeline:
          1. Preprocess frame
          2. HF Spaces API (LADI + victim + vehicles)
          3. Local YOLO (victim + fire/smoke + posture) in parallel
          4. Roboflow parallel models
          5. Groq verification
          6. Merge + return
        """
        processed = preprocess_for_detection(frame)
        img_bytes = encode_frame(processed, quality=88, max_dim=640)
        t0 = time.time()

        # ── Run everything in parallel ──
        results = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            f_hf      = ex.submit(self._post, "analyze/full", img_bytes)
            f_victim  = ex.submit(yolo_models.run_victim,     processed)
            f_fire    = ex.submit(yolo_models.run_fire_smoke, processed)
            f_posture = ex.submit(yolo_models.run_posture,    processed)
            f_rf      = ex.submit(rf_runner.run_all,          processed)

            results["hf"]      = f_hf.result()
            results["victim"]  = f_victim.result()
            results["fire"]    = f_fire.result()
            results["posture"] = f_posture.result()
            results["rf"]      = f_rf.result()

        hf = results["hf"]
        if "error" in hf:
            hf = {"results": {}, "zone_color": "green"}

        # ── Merge detections ──
        hf_victims = hf.get("results", {}).get("victims", {})
        hf_dets    = hf_victims.get("detections", [])
        local_dets = results["victim"].get("detections", [])

        # Deduplicate: keep higher confidence if boxes overlap significantly
        all_dets = _merge_detections(hf_dets, local_dets)

        classify_data   = hf.get("results", {}).get("classification", {})
        classifications = classify_data.get("top_predictions", [])
        fire_smoke      = results["fire"]
        posture         = results["posture"]

        # ── Groq verification ──
        verification = {}
        if verify and self.groq.enabled:
            verification = self.groq.verify(frame, all_dets, classifications,
                                            fire_smoke, posture)
            all_dets        = verification.get("verified_detections",      all_dets)
            classifications = verification.get("verified_classifications", classifications)

        # ── Triage summary ──
        triage = _compute_triage(all_dets)

        # ── Roboflow results ──
        rf_summary = {k: v["detections"] for k, v in results["rf"].items()}
        all_rf_dets = [d for dets in rf_summary.values() for d in dets]

        # ── Zone color ──
        has_fire = bool(fire_smoke) or verification.get("fire_smoke_confirmed", False)
        zone = _compute_zone_color(triage, all_rf_dets, has_fire)

        return {
            "zone_color":         zone,
            "detections":         all_dets,
            "classifications":    classifications,
            "triage_summary":     triage,
            "fire_smoke":         fire_smoke,
            "posture":            posture,
            "vehicles":           hf.get("results", {}).get("vehicles", {}),
            "roboflow":           rf_summary,
            "groq_accuracy":      verification.get("accuracy_score"),
            "groq_summary":       verification.get("groq_summary", ""),
            "groq_overrides":     verification.get("overrides_applied", 0),
            "critical_findings":  verification.get("critical_findings", ""),
            "total_time_ms":      round((time.time() - t0) * 1000, 2),
            "verified":           bool(verification),
            "timestamp":          time.time(),
        }

    def health(self):
        try:
            res = self.session.get(f"{self.base_url}/health", timeout=8)
            return res.json()
        except Exception as e:
            return {"error": str(e)}

    def start_keepalive(self, interval=600):
        def _ping():
            while True:
                try: self.session.get(f"{self.base_url}/health", timeout=5)
                except Exception: pass
                time.sleep(interval)
        threading.Thread(target=_ping, daemon=True, name="keepalive").start()
        print(f"💓 Keep-alive started ({interval//60}min)")


# ════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════
def _iou(b1, b2) -> float:
    """Intersection over union for two boxes."""
    xa = max(b1["xmin"], b2["xmin"]); ya = max(b1["ymin"], b2["ymin"])
    xb = min(b1["xmax"], b2["xmax"]); yb = min(b1["ymax"], b2["ymax"])
    inter = max(0, xb-xa) * max(0, yb-ya)
    a1 = (b1["xmax"]-b1["xmin"]) * (b1["ymax"]-b1["ymin"])
    a2 = (b2["xmax"]-b2["xmin"]) * (b2["ymax"]-b2["ymin"])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(list_a: list, list_b: list, iou_thresh=0.5) -> list:
    """
    Merge two detection lists. Where boxes overlap (IoU > threshold),
    keep the one with higher confidence.
    """
    merged = list(list_a)
    for det_b in list_b:
        box_b    = det_b.get("box", {})
        overlap  = False
        for i, det_a in enumerate(merged):
            box_a = det_a.get("box", {})
            if box_a and box_b and _iou(box_a, box_b) > iou_thresh:
                overlap = True
                if det_b.get("confidence", 0) > det_a.get("confidence", 0):
                    merged[i] = det_b
                break
        if not overlap:
            merged.append(det_b)
    merged.sort(key=lambda d: d.get("priority_score", d.get("confidence", 0)), reverse=True)
    return merged


def _compute_triage(detections: list) -> dict:
    if not detections:
        return {"total": 0, "critical": 0, "high": 0, "moderate": 0, "low": 0,
                "highest_score": 0.0, "action": "✅ No victims detected"}
    scored = []
    for d in detections:
        score = d.get("priority_score") or (
            d.get("confidence", 0.5) * CLASS_PRIORITY.get(d.get("class", ""), 0.5))
        rank  = ("CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.4
                 else "MODERATE" if score >= 0.2 else "LOW")
        scored.append({**d, "priority_score": round(score, 4), "priority_rank": rank})
    scored.sort(key=lambda d: d["priority_score"], reverse=True)
    c = sum(1 for d in scored if d["priority_rank"] == "CRITICAL")
    h = sum(1 for d in scored if d["priority_rank"] == "HIGH")
    m = sum(1 for d in scored if d["priority_rank"] == "MODERATE")
    l = sum(1 for d in scored if d["priority_rank"] == "LOW")
    action = ("⚠️ IMMEDIATE RESCUE — Critical victims" if c else
              "🔴 Deploy rescue team" if h else
              "🟡 Triage required" if m else "🟢 Low priority")
    return {"total": len(scored), "critical": c, "high": h, "moderate": m, "low": l,
            "highest_score": scored[0]["priority_score"] if scored else 0.0,
            "action": action}


def _compute_zone_color(triage: dict, rf_dets: list, has_fire: bool) -> str:
    if has_fire or triage.get("critical", 0) > 0:
        return "red"
    if triage.get("high", 0) > 0:
        return "orange"
    if triage.get("total", 0) > 0 or rf_dets:
        return "yellow"
    return "green"


# ════════════════════════════════════════════════════════
# ANNOTATION — draw boxes on frame (returned as JPEG)
# ════════════════════════════════════════════════════════
RANK_COLORS = {
    "CRITICAL": (0,0,255), "HIGH": (0,165,255),
    "MODERATE": (0,255,255), "LOW": (0,200,0),
}
TYPE_COLORS = {
    "fire": (0,80,255), "smoke": (120,120,200), "flame": (0,60,255),
    "standing": (200,200,0), "lying_down": (0,100,255),
    "collapsed": (0,0,220), "waving": (255,200,0),
}
RF_COLORS = {
    "rf_custom_1": (255,100,0), "rf_custom_2": (0,255,150),
    "rf_custom_3": (200,0,255), "rf_custom_4": (255,255,0),
}

def annotate_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()

    # Victim detections
    for d in result.get("detections", []):
        box = d.get("box")
        if not box: continue
        x1,y1,x2,y2 = box["xmin"],box["ymin"],box["xmax"],box["ymax"]
        color = RANK_COLORS.get(d.get("priority_rank", "LOW"), (180,180,180))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        tag   = "[G]" if d.get("groq_action") == "OVERRIDDEN" else ""
        label = f"{d.get('class','?')} {d.get('confidence',0):.2f}{tag}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out,(x1,y1-18),(x1+tw+4,y1),color,-1)
        cv2.putText(out,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)

    # Fire & smoke
    for d in result.get("fire_smoke", []):
        box = d.get("box")
        if not box: continue
        x1,y1,x2,y2 = box["xmin"],box["ymin"],box["xmax"],box["ymax"]
        color = TYPE_COLORS.get(d.get("class","fire"), (0,80,255))
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f"🔥{d.get('class','?')} {d.get('confidence',0):.2f}",
                    (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)

    # Posture
    for d in result.get("posture", []):
        box = d.get("box")
        if not box: continue
        x1,y1,x2,y2 = box["xmin"],box["ymin"],box["xmax"],box["ymax"]
        color = TYPE_COLORS.get(d.get("class","standing"),(200,200,0))
        cv2.rectangle(out,(x1,y2),(x2,y2+20),color,-1)
        cv2.putText(out,f"{d.get('class','?')} risk:{d.get('risk_score',0):.2f}",
                    (x1+2,y2+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)

    # Roboflow custom models
    rf_data = result.get("roboflow", {})
    for key, dets in rf_data.items():
        color = RF_COLORS.get(key, (255,255,255))
        for d in dets:
            box = d.get("box")
            if not box: continue
            x1,y1,x2,y2 = box["xmin"],box["ymin"],box["xmax"],box["ymax"]
            cv2.rectangle(out,(x1,y1),(x2,y2),color,1)
            cv2.putText(out,f"[RF]{d.get('class','?')}",(x1,y2+12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)

    # Zone banner
    zone  = result.get("zone_color", "green")
    bclr  = {"red":(0,0,200),"orange":(0,140,255),"yellow":(0,200,220),"green":(0,170,0)}
    cv2.rectangle(out,(0,0),(out.shape[1],34),bclr.get(zone,(100,100,100)),-1)
    acc = result.get("groq_accuracy")
    txt = (f"ZONE:{zone.upper()}  Groq:{acc}/10" if acc else f"ZONE:{zone.upper()}")
    txt += f"  {datetime.now().strftime('%H:%M:%S')}"
    cv2.putText(out,txt,(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)

    # Critical findings at bottom
    cf = result.get("critical_findings","")
    if cf:
        cv2.putText(out,f"⚠ {cf[:80]}",(6,out.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,200,255),1)

    return out


def frame_to_b64(frame: np.ndarray, quality=80) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ════════════════════════════════════════════════════════
# SERVER STATE
# ════════════════════════════════════════════════════════
class ServerState:
    def __init__(self):
        self._lock    = threading.Lock()
        self.rover    = {"lat": 0.0, "lng": 0.0, "heading": 0.0,
                         "speed": 0.0, "mode": "manual"}
        self.objects  = {}
        self.path     = []
        self.last_result     = {}
        self.last_annotated  = ""  # base64 annotated frame

    def update_rover(self, data):
        with self._lock:
            self.rover.update({k: v for k, v in data.items() if k in self.rover})

    def add_object(self, obj):
        oid = obj.get("id") or str(uuid.uuid4())[:8]
        with self._lock: self.objects[oid] = {**obj, "id": oid}
        return oid

    def remove_object(self, oid):
        with self._lock: self.objects.pop(oid, None)

    def set_path(self, path):
        with self._lock: self.path = path

    def set_result(self, result, annotated_b64=""):
        with self._lock:
            self.last_result    = result
            self.last_annotated = annotated_b64

    def snapshot(self):
        with self._lock:
            return {"rover": dict(self.rover), "objects": dict(self.objects),
                    "path": list(self.path), "last_result": dict(self.last_result),
                    "last_annotated": self.last_annotated}


# ════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════
flask_app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(flask_app)

state   = ServerState()
planner = PathPlanner(grid_resolution=1.0)
client  = None


@flask_app.route("/")
def dashboard():
    return render_template("dashboard.html")


@flask_app.route("/api/health")
def api_health():
    hf = client.health() if client else {"error": "not initialized"}
    return jsonify({
        "server":        "ok",
        "hf_api":        hf,
        "groq":          client.groq.enabled if client else False,
        "local_models":  yolo_models.status(),
        "roboflow":      {k: v.get("enabled", False) for k, v in ROBOFLOW_MODELS.items()},
    })


@flask_app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(state.snapshot())


@flask_app.route("/api/state", methods=["POST"])
def update_state():
    state.update_rover(request.get_json(silent=True) or {})
    return jsonify({"ok": True})


@flask_app.route("/api/objects", methods=["GET"])
def get_objects():
    return jsonify({"objects": list(state.objects.values())})


@flask_app.route("/api/objects", methods=["POST"])
def add_object():
    data = request.get_json(silent=True) or {}
    if not {"type","lat","lng"}.issubset(data.keys()):
        return jsonify({"error": "Missing type/lat/lng"}), 400
    oid = state.add_object({
        "id": data.get("id"), "type": data["type"],
        "lat": float(data["lat"]), "lng": float(data["lng"]),
        "radius": float(data.get("radius", 2.0)),
        "label": data.get("label", data["type"]),
    })
    return jsonify({"id": oid, "ok": True})


@flask_app.route("/api/objects/<oid>", methods=["DELETE"])
def delete_object(oid):
    state.remove_object(oid)
    return jsonify({"ok": True})


@flask_app.route("/api/objects/clear", methods=["POST"])
def clear_objects():
    with state._lock: state.objects.clear()
    return jsonify({"ok": True})


@flask_app.route("/api/pathplan", methods=["POST"])
def api_pathplan():
    data         = request.get_json(silent=True) or {}
    start_gps    = data.get("start") or state.rover
    all_objects  = list(state.objects.values())
    targets      = data.get("targets")   or [o for o in all_objects if o["type"] in ("victim","checkpoint")]
    obstacles    = data.get("obstacles") or [o for o in all_objects if o["type"] in ("debris","hole","bump")]
    if not targets:
        return jsonify({"error": "No target/victim points defined"}), 400
    result = planner.plan(start_gps, targets, obstacles)
    state.set_path(result["path"])
    return jsonify(result)


@flask_app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not client:
        return jsonify({"error": "AI client not initialized"}), 503
    data = request.get_json(silent=True) or {}
    b64  = data.get("frame", "")
    if not b64:
        return jsonify({"error": "No frame provided"}), 400
    try:
        if "," in b64: b64 = b64.split(",",1)[1]
        nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Cannot decode frame"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    result       = client.analyze(frame, verify=True)
    annotated    = annotate_frame(frame, result)
    annotated_b64= frame_to_b64(annotated, quality=75)
    result["annotated_frame"] = annotated_b64
    state.set_result(result, annotated_b64)
    if data.get("lat") and data.get("lng"):
        state.update_rover({"lat": data["lat"], "lng": data["lng"]})
    return jsonify(result)


@flask_app.route("/api/stream")
def api_stream():
    def events():
        while True:
            snap = state.snapshot()
            snap.pop("last_annotated", None)  # don't send large b64 via SSE
            yield f"data: {json.dumps(snap)}\n\n"
            time.sleep(1)
    return Response(events(), mimetype="text/event-stream")


# ════════════════════════════════════════════════════════
# CLI UTILITIES
# ════════════════════════════════════════════════════════
def print_result(data):
    if not data or "error" in data:
        print(f"❌ {data.get('error','No result')}"); return
    zone  = data.get("zone_color","green").upper()
    icons = {"RED":"🔴","ORANGE":"🟠","YELLOW":"🟡","GREEN":"🟢"}
    print(f"\n{'═'*55}")
    print(f"{icons.get(zone,'⚪')} ZONE:{zone}  ⏱{data.get('total_time_ms')}ms")
    if data.get("groq_accuracy"):
        print(f"🧠 Groq: {data['groq_accuracy']}/10 | overrides:{data.get('groq_overrides',0)}")
    if data.get("groq_summary"):
        print(f'   "{data["groq_summary"]}"')
    if data.get("critical_findings"):
        print(f"   ⚠️  {data['critical_findings']}")
    t = data.get("triage_summary",{})
    if t.get("total",0):
        print(f"\n👥 Victims:{t.get('total',0)} | Critical:{t.get('critical',0)} | {t.get('action','-')}")
    if data.get("fire_smoke"):
        print(f"🔥 Fire/Smoke: {len(data['fire_smoke'])} detections")
    if data.get("posture"):
        for p in data["posture"][:2]:
            print(f"   posture:{p['class']} risk:{p['risk_score']:.2f}")
    rf = {k: len(v) for k,v in data.get("roboflow",{}).items() if v}
    if rf: print(f"📡 Roboflow: {rf}")
    print(f"{'═'*55}\n")


def run_image(path):
    frame = cv2.imread(path)
    if frame is None: print(f"❌ Cannot read: {path}"); return
    result    = client.analyze(frame)
    print_result(result)
    annotated = annotate_frame(frame, result)
    out_path  = f"result_{os.path.basename(path)}"
    cv2.imwrite(out_path, annotated)
    print(f"💾 {out_path}")
    cv2.imshow("Disaster AI", annotated); cv2.waitKey(0); cv2.destroyAllWindows()


def run_webcam(cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened(): print(f"❌ Cannot open webcam {cam}"); return
    last_result = None; last_send = 0; pending = False

    def send(frame):
        nonlocal last_result, pending
        pending     = True
        last_result = client.analyze(frame.copy())
        print_result(last_result)
        pending     = False

    print("📷 Live — Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        now = time.time()
        if now - last_send >= 3 and not pending:
            last_send = now
            threading.Thread(target=send, args=(frame,), daemon=True).start()
        disp = annotate_frame(frame, last_result) if last_result else frame
        if pending:
            cv2.putText(disp,"Analyzing...",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,200),1)
        cv2.imshow("Disaster AI", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster AI v3")
    parser.add_argument("--url",    default=HF_API_URL)
    parser.add_argument("--groq",   default=GROQ_API_KEY)
    parser.add_argument("--serve",  action="store_true")
    parser.add_argument("--port",   type=int, default=5050)
    parser.add_argument("--image",  help="Image file path")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--cam",    type=int, default=0)
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🚀 Disaster AI v3.0")
    print("="*50)

    # Load local YOLO models
    yolo_models.load_all()

    # Init client
    client = DisasterAIClient(base_url=args.url, groq_key=args.groq or GROQ_API_KEY)
    client.start_keepalive()

    if args.health:
        print(json.dumps(client.health(), indent=2))
    elif args.image:
        run_image(args.image)
    elif args.webcam:
        run_webcam(args.cam)
    elif args.serve:
        print(f"\n🌐 Dashboard → http://localhost:{args.port}")
        flask_app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    else:
        parser.print_help()
