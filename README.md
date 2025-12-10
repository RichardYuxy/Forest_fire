# Forest_fire
下面为你基于 **app.py** 与 **Detection_and_calculate.py** 的内容撰写了一份 **专业、完整、可直接用于 GitHub 的 README.md**。
（已经根据文件内容自动分析了接口、依赖、功能、输入输出、运行方法）

---

# 🌲 Wildfire Damage Detection System

### **Remote Sensing Burned-Area Segmentation & Damage Calculation API**

本项目包含两个核心模块：

1. **app.py — FastAPI 推理服务**
   提供图像上传、下载、模型推理、结果返回的 HTTP API
2. **Detection_and_calculate.py — 火烧迹地面积计算脚本**
   对分割结果进行红色像素统计、地面覆盖换算和面积估计

该系统可用于森林火灾、林草地烧毁检测等任务，可部署在服务器端对无人机/卫星影像进行分割分析。

---

# 📌 目录

* [功能概述](#功能概述)
* [项目结构](#项目结构)
* [环境依赖](#环境依赖)
* [FastAPI 服务说明](#fastapi-服务说明)
* [推理流程](#推理流程)
* [面积计算脚本说明](#面积计算脚本说明)
* [示例输入输出](#示例输入输出)
* [运行方式](#运行方式)

---

# 功能概述

### **1. app.py – 云端模型推理 API**

* 基于 **FastAPI**
* 支持 **本地文件上传** 或 **URL 下载图像**
* 自动调用预加载的 **UNet/UNet++ 分割模型**
* 返回：

  * 分割结果图（以红色区域表示烧毁地）
  * JSON 格式统计数据（像素计数、占比等）
* 所有图像按 **Mission ID** 分类存储（uploads/, results/）

---

### **2. Detection_and_calculate.py – 面积估算**

对推理得到的分割图（通常红色掩膜）执行：

* 红色像素计数
* 红色比例 = 红色像素数 / 总像素
* 根据地面分辨率（用户可设定）换算面积
* 输出：

  * 烧毁面积（m²）
  * 红色占比 %
  * 地面覆盖估计面积

---

# 📁 项目结构

```
.
├── app.py                    # FastAPI 服务器：图像上传、下载、推理、返回结果
├── Detection_and_calculate.py# 红色像素统计与烧毁面积估计
├── models/                   # 分割模型 (*.pth)
├── uploads/                  # 上传原图按 mission_id 存储
├── results/                  # 推理输出图像
└── README.md
```

---

# 🧩 环境依赖

### Python 版本

```
Python ≥ 3.8
```

### 安装依赖

```
pip install fastapi uvicorn opencv-python pillow numpy torch torchvision requests
```

---

# 🚀 FastAPI 服务说明（app.py）

### 启动服务器

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📌 API 接口

### **1. /upload — 上传图像**

| 方法   | 描述                             |
| ---- | ------------------------------ |
| POST | 上传图片（multipart/form-data）并触发分割 |

**参数**

* `mission_id`: 任务 ID
* `file`: 图像文件

返回：

```json
{
  "mission_id": "001",
  "input_image": "uploads/001/img1.jpg",
  "result_image": "results/001/img1_mask.png",
  "red_ratio": 0.34
}
```

---

### **2. /download — 从 URL 下载图像并推理**

输入 JSON：

```json
{
  "mission_id": "002",
  "url": "https://example.com/a.jpg"
}
```

返回与 `/upload` 相同。

---

### **3. /health — 健康检查**

```json
{ "status": "ok" }
```

---

# 🔍 推理流程（app.py 内部逻辑）

1. **读取图像** → 转为 RGB
2. **进入模型预处理**：

   * resize
   * normalization
   * tensor
3. **模型推理**
4. **输出 mask**（红色区域为烧毁地）
5. **计算红色像素比例**
6. **保存 mask 到 results/mission_id/**
7. 返回 JSON + 可视化结果

---

# 🔥 面积计算脚本说明（Detection_and_calculate.py）

此脚本用于对推理输出的红色 mask 图像执行面积计算。

### 使用方式

```
python Detection_and_calculate.py path/to/mask.png
```

### 核心步骤

* 用 OpenCV 读取分割图
* 识别 **纯红像素 (B=0,G=0,R=255)**
* 统计红色数量与总像素
* 红色面积：

```
burned_area = red_pixel_count × ground_resolution_m²
```

脚本默认输出示例：

```
红色像素数 = 105393
总像素 = 512000
红色比例 ≈ 20.58%

假设地面分辨率 1 像素=0.04 m^2
→ 火烧迹地估计面积 ≈ 4215 m^2
（整幅图地面覆盖≈ 20480 m^2；红色像素占比≈ 20.58%）
```

---

# 📸 示例输入输出

输入图：
（原始无人机 / 卫星影像）

模型输出：

* 红色区域=烧毁地
* 白色/黑色区域=未烧毁

JSON 输出示例：

```json
{
  "mission_id": "forest_003",
  "red_pixels": 83241,
  "total_pixels": 300000,
  "red_ratio": 0.277,
  "estimated_area_m2": 3150
}
```

---

# 🏃 如何运行项目

## 1. 启动推理服务

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 2. 上传图像

```
POST /upload
```

## 3. 调用面积计算

```
python Detection_and_calculate.py results/<mission_id>/<filename>
```

---

# 🧭 后续可扩展功能（可写进你的论文/报告）

* 多类别火灾等级判定
* GPS → 地面分辨率自动换算
* 多模型集成（UNet++、DeepLab、SegFormer）
* FastAPI + MinIO 分布式存储
* WebSocket 实时推理
* Docker 部署

---

如果你需要我 **帮你生成真正完整的 GitHub README.md 文件（含标题图、接口示意图、文件结构图、示例图）**，或想把 README 转成 **中英双语版**，我也可以帮你做。
