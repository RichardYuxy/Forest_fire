# app.py
import os, io, re, glob, traceback, mimetypes, hmac, hashlib, json, time, shutil, zipfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import cv2
from PIL import Image, ImageOps
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from pydantic import BaseModel, AnyUrl, Field
import requests
import httpx
from minio import Minio

# ===== 你自己的算法模块 =====
from Detection_and_calculate import (
    predict_mask,
    load_segmentation_model,
    calculate_red_area_image,
    merge_mask_tiles_to_png, network_1_prediction, generate_crops, network_2_predict_single
)

SEG_MODEL = None

# ========= 简易“数据库”：可替换为真实持久化 =========
EVALUATIONS_DB: Dict[int, Dict[str, Any]] = {}
_AUTO_ID = 1

def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()

def _next_id() -> int:
    global _AUTO_ID
    v = _AUTO_ID
    _AUTO_ID += 1
    return v

def seed_evaluation_record(mission_id: int):
    """在任务启动时建单（进行中）。真实项目：INSERT 到 DB。"""
    if mission_id not in EVALUATIONS_DB:
        EVALUATIONS_DB[mission_id] = {
            "id": _next_id(),
            "missionId": mission_id,
            "assessmentStatus": 0,        # 0=进行中, 1=已完成
            "statusDescription": "进行中",
            "createdAt": _now_iso(),
            "assessedAt": None,
            "maskImageUrl": None,
            "burnedArea": None,
            "economicLoss": None,
        }

def _update_record_done(
    mission_id: int,
    mask_url: Optional[str],
    burned_area_m2: float,
    economic_loss: float,
):
    rec = EVALUATIONS_DB.get(mission_id)
    if not rec:
        seed_evaluation_record(mission_id)
        rec = EVALUATIONS_DB[mission_id]
    rec.update({
        "maskImageUrl": mask_url,
        "burnedArea": float(burned_area_m2),
        "economicLoss": float(economic_loss),
        "assessmentStatus": 1,
        "statusDescription": "已完成",
        "assessedAt": _now_iso(),
    })
    if not rec.get("createdAt"):
        rec["createdAt"] = _now_iso()
    return rec

# ========= MinIO 工具 =========
def _minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9001")
    access   = os.getenv("MINIO_ACCESS_KEY", "minio")
    secret   = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    secure   = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)

def _ensure_bucket(client: Minio, bucket: str):
    found = client.bucket_exists(bucket)
    if not found:
        client.make_bucket(bucket)

def _guess_content_type(path: str, fallback: str = "application/octet-stream"):
    ct, _ = mimetypes.guess_type(path)
    return ct or fallback

async def upload_to_minio(local_path: str, object_name: str, presign: bool = True) -> str:
    """
    把本地文件上传到 MinIO 并返回 URL。
    - 若桶是公共读：presign=False，返回直链 URL；
    - 若桶是私有：presign=True，返回预签名 GET URL（有效期由 MINIO_PRESIGN_TTL 控制）。
    """
    client = _minio_client()
    bucket = os.getenv("MINIO_BUCKET", "disaster")
    await run_in_threadpool(_ensure_bucket, client, bucket)

    content_type = _guess_content_type(local_path, "application/octet-stream")

    def _put():
        with open(local_path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            client.put_object(bucket, object_name, f, size, content_type=content_type)
    await run_in_threadpool(_put)

    # 构造 URL
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    scheme = "https" if secure else "http"
    endpoint = os.getenv("MINIO_ENDPOINT", "10.112.247.:9001")
    public_url = f"{scheme}://{endpoint}/{bucket}/{object_name}"

    # 私有桶 → 预签名
    if presign:
        ttl = int(os.getenv("MINIO_PRESIGN_TTL", "86400"))  # 24h
        def _presign():
            return client.get_presigned_url(
                "GET", bucket, object_name, expires=timedelta(seconds=ttl)
            )
        return await run_in_threadpool(_presign)

    return public_url

# ========= 回调后端（带重试/可选鉴权/HMAC） =========
def _make_headers(body_bytes: bytes) -> dict:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("CALLBACK_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    secret = os.getenv("CALLBACK_SECRET", "").strip()
    if secret:
        sig = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        headers["X-Signature"] = sig
    return headers

async def send_callback_reliable(mission_id: int, mask_url: str, burned_area: float, economic_loss: float, task_id: str):
    cb_url = os.getenv("CALLBACK_URL", "").strip()
    if not cb_url:
        return False

    payload = {
        "missionId": mission_id,
        "maskImageUrl": mask_url,
        "burnedArea": float(burned_area),
        "economicLoss": float(economic_loss),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _make_headers(body)

    max_retries = int(os.getenv("CALLBACK_MAX_RETRIES", "3"))
    backoff = 1.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(cb_url, content=body, headers=headers)
                if 200 <= resp.status_code < 300:
                    return True
                else:
                    print(f"[callback] attempt {attempt} status={resp.status_code} body={resp.text}")
            except Exception as e:
                print(f"[callback] attempt {attempt} error={e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 16.0)
    return False


# ========= FastAPI app & 生命周期（预热模型） =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEG_MODEL
    try:
        if load_segmentation_model is not None:
            SEG_MODEL = load_segmentation_model()   # 启动时预热 -> 热启动
            print("[lifespan] segmentation model preloaded.")
        else:
            print("[lifespan] no load_segmentation_model; will lazy-load inside predict.")
    except Exception as e:
        SEG_MODEL = None
        print(f"[lifespan] model preload skipped: {e}")
    yield
    # 可选清理

app = FastAPI(title="Fire Damage Inference API", version="1.3", lifespan=lifespan)

# CORS + 静态目录
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("labels", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")
app.mount("/labels", StaticFiles(directory="labels"), name="labels")

# ========= 健康检查 =========
@app.get("/health")
def health():
    try:
        import tensorflow as tf
        gpu = str(tf.config.list_physical_devices("GPU"))
        tfv = tf.__version__
    except Exception as e:
        gpu = "unknown"
        tfv = f"tf-not-available: {e}"
    return {"status": "ok", "tensorflow": tfv, "gpu": gpu}

# ========= 请求体模型 =========
class EvalRequest(BaseModel):
    missionId: int = Field(..., ge=1)
    synthesizedImageUrl: AnyUrl
    realWorldWidthMeters: float = Field(..., gt=0)
    realWorldHeightMeters: float = Field(..., gt=0)

class EvalCallbackRequest(BaseModel):
    missionId: int = Field(..., ge=1, description="任务ID")
    maskImageUrl: AnyUrl = Field(..., description="掩码结果URL（此处为ZIP包URL，因为不再合并为单张）")
    burnedArea: float = Field(..., gt=0, description="过火面积（平方米）")
    economicLoss: float = Field(..., ge=0, description="经济损失（元）")

# ========= 任务启动接口（JSON 输入 → 分割 → 统计 → ZIP→ MinIO → 回调） =========
@app.post("/api/algorithm/generate-evaluation")
async def generate_evaluation(payload: EvalRequest, background: BackgroundTasks):
    """
    输入 JSON：
    {
      "missionId": 1001,
      "synthesizedImageUrl": "https://oss.example.com/synthesis/syn_1001.jpg",
      "realWorldWidthMeters": 1000.0,
      "realWorldHeightMeters": 800.0
    }
    返回（成功）：
    {
      "code": 200,
      "message": "灾损评估任务已启动",
      "data": { "taskId": "...", "missionId": 1001, "status": "success" }
    }
    """
    try:
        mission_id = payload.missionId
        # 1) 建单 seed
        seed_evaluation_record(mission_id)

        # 2) 面积
        footprint_area_m2 = float(payload.realWorldWidthMeters) * float(payload.realWorldHeightMeters)

        # 3) 下载合成图
        def _download():
            r = requests.get(str(payload.synthesizedImageUrl), timeout=(5, 30))
            r.raise_for_status()
            return r.content
        try:
            raw = await run_in_threadpool(_download)
        except Exception as e:
            return JSONResponse(
                {"code": 400, "message": f"合成图片URL无效或参数错误: {e}", "data": None},
                status_code=400
            )

        # 4) 保存到 uploads
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        img_path = os.path.join("uploads", f"mission_{mission_id}.jpg")
        pil.save(img_path, "JPEG", quality=92)

        # 5）裁剪
        def _run_crop():
            # 约定：generate_crops(image_path, output_dir, window_size=228, stride=114, resize_to=(912, 912))
            # 若 generate_crops 返回裁剪文件列表，直接返回；若只落盘，这里补一层收集。
            out_dir = f"uploads/crops_{mission_id}"
            ret = generate_crops(img_path, out_dir)  # 你的函数签名里后三个参数都有默认值，这里不用传
            if isinstance(ret, (list, tuple)) and len(ret) > 0:
                return sorted(ret)
            else:
                import glob
                return sorted(glob.glob(os.path.join(out_dir, "*.*")))

        crop_files = await run_in_threadpool(_run_crop)
        if not crop_files:
            return JSONResponse({"code": 400, "message": "裁剪阶段未产生任何裁剪图块", "data": None}, status_code=400)

        def _run_predict():
            label_dir = f"uploads/label_{mission_id}"
            os.makedirs(label_dir, exist_ok=True)

            if SEG_MODEL is not None:
                models = SEG_MODEL
                for i, crop_path in enumerate(crop_files):
                    pred_mask = network_2_predict_single(crop_path, model_l2=models.get("l2"))

                    # 二值化到 0/255（uint8）
                    mask_bin = (pred_mask.astype(np.uint8)) * 255

                    mask_name = f"mask_{os.path.basename(crop_path)}"
                    mask_path = os.path.join(label_dir, mask_name)

                    # 确保父目录存在（健壮性）
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

                    cv2.imwrite(mask_path, mask_bin)

                    if (i + 1) % 50 == 0 or i == len(crop_files) - 1:
                        print(f"[{i + 1}/{len(crop_files)}] 保存掩码至: {mask_path}")

                print(f"✅ 所有掩码已保存至: {label_dir}")
                return label_dir
            else:
                out_dir = predict_mask(img_path, "labels")
                return out_dir
        label_dir = await run_in_threadpool(_run_predict)

        def _run_merge():
            return merge_mask_tiles_to_png(
                label_dir,
                f"uploads/merged_{mission_id}.png",
                window_size=228,
                stride=114,
                resize_to=(912, 912),
                reference_image=f"uploads/mission_{mission_id}.jpg"
            )
        merged_image = await run_in_threadpool(_run_merge)

        # 6) 统计比例 → 过火面积（不合并，直接基于裁剪目录）
        def _run_stats():
            return calculate_red_area_image(merged_image, footprint_area_m2)
        red_ratio, burned_area_m2 = await run_in_threadpool(_run_stats)

        # 7) 经济损失估算（线性单价，可通过环境变量 ECON_LOSS_PER_M2 配置；默认 0）
        loss_coef = float(os.getenv("ECON_LOSS_PER_M2", "20.0"))
        economic_loss = burned_area_m2 * loss_coef


        # 9) 上传 MinIO，得到 URL（作为 maskImageUrl 返回/回调）
        mask_obj_key = merged_image
        mask_minio_url = await upload_to_minio(f"uploads/merged_{mission_id}.png", mask_obj_key, presign=True)

        # 10) 更新记录（保存 MinIO URL）
        rec = _update_record_done(
            mission_id=mission_id,
            mask_url=mask_minio_url,
            burned_area_m2=burned_area_m2,
            economic_loss=economic_loss,
        )

        # 11) 生成任务 ID
        task_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 12) 在后台调用回调（不阻塞当前返回）
        background.add_task(
            send_callback_reliable,
            mission_id, mask_minio_url, burned_area_m2, economic_loss, task_id
        )

        # 13) 标准成功返回
        return JSONResponse(
            {
                "missionId": mission_id,
                "maskImageUrl": mask_minio_url,
                "burnedArea": round(burned_area_m2, 4),
                "economicLoss": round(economic_loss, 2)
            },
            status_code=200
        )

    except Exception as e:
        tb = traceback.format_exc()
        print("[generate-evaluation error]", tb)
        return JSONResponse(
            {"code": 400, "message": f"合成图片URL无效或参数错误: {e}", "data": None},
            status_code=400
        )

# ========= 回调接口（后端调用，用于最终入库） =========
@app.post("/admin/evaluation/callback")
async def evaluation_callback(payload: EvalCallbackRequest):
    """
    请求示例：
    {
      "missionId": 1001,
      "maskImageUrl": "https://minio.../masks/mission_1001_mask_tiles.png",
      "burnedArea": 15600.5,
      "economicLoss": 320000.0
    }
    """
    try:
        mission_id = payload.missionId
        rec = EVALUATIONS_DB.get(mission_id)
        if not rec:
            return JSONResponse(
                {
                    "error": "处理算法回调失败: 未找到对应的评估记录",
                    "timestamp": _now_iso(),
                },
                status_code=404,
            )

        rec = _update_record_done(
            mission_id=mission_id,
            mask_url=str(payload.maskImageUrl),
            burned_area_m2=float(payload.burnedArea),
            economic_loss=float(payload.economicLoss),
        )

        return {
            "id": rec["id"],
            "missionId": rec["missionId"],
            "maskImageUrl": rec["maskImageUrl"],
            "burnedArea": rec["burnedArea"],
            "economicLoss": rec["economicLoss"],
            "assessmentStatus": rec["assessmentStatus"],
            "statusDescription": rec["statusDescription"],
            "createdAt": rec["createdAt"],
            "assessedAt": rec["assessedAt"],
        }

    except Exception as e:
        return JSONResponse(
            {"error": f"处理算法回调失败: {e}", "timestamp": _now_iso()},
            status_code=400,
        )

# ========= main =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
