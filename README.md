# 🔥 Wildfire Damage Evaluation Algorithm Module

### **灾损评估算法模块 README**

本项目实现了灾损评估系统中的 **“灾损评估算法模块（核心）”**。
模块通过标准接口接收任务信息，对图像进行分割生成火烧迹地掩码，并计算过火面积与经济损失，最终通过回调接口向主系统返回评估结果。

本 README 仅包含你需要在部署时对接系统的关键信息。

---

# 📁 目录

* [1. 模块简介](#模块简介)
* [2. 文件结构](#文件结构)
* [3. 环境依赖](#环境依赖)
* [4. 服务启动方式](#服务启动方式)
* [6. 灾损评估算法模块接口（核心）](#灾损评估算法模块接口核心)
* [7. 面积计算逻辑](#面积计算逻辑)
* [8. 关键回调接口](#关键回调接口)
* [9. 完整业务流程](#完整业务流程)
* [10. 注意事项](#注意事项)

---

# 1. 模块简介

本项目负责以下工作：

### 🔥 **灾损评估（核心功能）**

* 下载合成图（synthesizedImageUrl）
* 使用深度学习模型进行分割（生成 mask）
* 使用 `Detection_and_calculate.py` 计算：

  * 红色像素比例
  * 过火面积（㎡）
  * 经济损失（元）
* 上传掩码图
* 回调主系统 `/admin/evaluation/callback`

所有字段、接口命名和流程均遵循系统接口规范。

---

# 2. 文件结构

```
.
├── app.py                      # FastAPI 服务，实现对接 API
├── Detection_and_calculate.py  # 掩码统计 + 面积计算脚本
├── models/                     # wildfire segmentation model
├── results/                    # 输出掩码图
├── uploads/                    # 下载的输入图
└── README.md
```

---

# 3. 环境依赖

```
pip install -r requirements.txt
```

---

# 4. 服务启动方式

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

启动后模块将暴露：

| 类型 | API                                  |
| -- | ------------------------------------ |
| 核心 | `/api/algorithm/generate-evaluation` |

---

# 5. 灾损评估算法模块接口（核心）

你的 **app.py** 的主要任务。

---

## 📌 POST /api/algorithm/generate-evaluation

系统在完成图片合成回调后自动调用。

### 请求内容（来自系统）

```json
{
  "missionId": 1001,
  "synthesizedImageUrl": "https://oss.example.com/synthesis/syn_1001.jpg",
  "realWorldWidthMeters": 1000.0,
  "realWorldHeightMeters": 800.0
}
```

### 响应格式
**成功响应:**
```json
{
  "code": 200,
  "message": "灾损评估任务已启动",
  "data": {
    "taskId": "eval_20251101_001",
    "missionId": 1001,
    "status": "success"
  }
}
```
**失败响应**
```json
{
  "code": 400,
  "message": "合成图片URL无效或参数错误",
  "data": null
}
```

---

## 模块处理步骤（与 app.py 一致）

1. 解析参数
2. 下载合成图
3. 执行分割模型 → 得到 mask
4. 保存 mask 到：

   ```
   results/<missionId>/mask_xxx.png
   ```
5. 使用 Detection_and_calculate.py 计算面积
6. 计算经济损失（可自定义）
7. 上传掩码图
8. 调用主系统回调： `/admin/evaluation/callback`

---

## 📌 回调内容（必须与系统规范一致）
###  请求示例
```json
{
  "missionId": 1001,
  "maskImageUrl": "https://your-host/masks/mask_1001.png",
  "burnedArea": 15600.5,
  "economicLoss": 320000.0
}
```
### 响应示例
**成功响应**
```json
{
  "id": 1,
  "missionId": 1001,
  "maskImageUrl": "https://oss.example.com/masks/mask_1001.png",
  "burnedArea": 15600.5,
  "economicLoss": 320000.0,
  "assessmentStatus": 1,
  "statusDescription": "已完成",
  "createdAt": "2024-10-20T10:00:00",
  "assessedAt": "2024-10-20T11:30:00"
}
```
**失败响应**
```json
{
  "error": "处理算法回调失败: 未找到对应的评估记录",
  "timestamp": "2024-10-20T11:30:00"
}
```

---

# 6. 面积计算逻辑

由 `Detection_and_calculate.py` 完成：

### 输入

* 掩码图片（红色区域表示烧毁区域）
* 真实世界宽度、高度（米）

### 输出内容示例

```
红色像素 = 103920
总像素 = 512000
红色比例 = 20.29%
过火面积 = 15600.5 m^2
```

### 面积换算公式

```
pixel_size_width  = realWorldWidthMeters / image_width
pixel_size_height = realWorldHeightMeters / image_height

pixel_real_area = pixel_size_width * pixel_size_height

burnedArea = red_pixels * pixel_real_area
```

经济损失可根据业务规则自定义。

---

# 7. 关键回调接口

---

## 🔄 灾损评估回调

`POST /admin/evaluation/callback`

```json
{
  "missionId": 1001,
  "maskImageUrl": "https://your-host/masks/mask_1001.png",
  "burnedArea": 15600.5,
  "economicLoss": 320000.0
}
```

---

# 9. 完整业务流程

```
系统触发评估
   ↓
系统自动调用灾损评估模块
POST /api/algorithm/generate-evaluation
   ↓
模型推理 + 掩码生成 + 面积计算
   ↓
POST /admin/evaluation/callback
```

流程来自互联模块文档规范。

---

# 10. 注意事项

* 所有 URL 必须是可公网访问的（OSS / 本地反向代理）
* 掩码建议为 PNG（保持颜色一致）
* 红色像素需使用统一标准：`(R=255, G=0, B=0)`
* 回调失败需重试或记录日志
* missionId 必须原样返回
