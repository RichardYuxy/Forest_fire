import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL.ExifTags import GPSTAGS, TAGS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------- 2. 预测掩码模块（来自predict.py，简化适配单张图像） --------------------------
# 模型参数
IMG_HEIGHT = 912
IMG_WIDTH = 912
CROP_HEIGHT = 228
CROP_WIDTH = 228
BURNED_PIXEL_VALUE = 1
PATH_WEIGHT_NETWORK_1 = "/Users/xiaoyu/Downloads/network_1_weights/checkpoint"  # 替换为实际模型权重路径
PATH_WEIGHT_NETWORK_2 = "/Users/xiaoyu/Downloads/network_2_weights/checkpoint"  # 替换为实际模型权重路径

# ===== 模型定义与一次性加载 =====
from models import unetpp_level_1, unet_level_2  # 需确保models.py存在

_SEG_MODELS = {"l1": None, "l2": None}  # 模块级单例


def load_model_weights(model, path_weight: str):
    """
    仅加载权重到结构相同的Keras模型。
    若你改成SavedModel，请替换为：
        tf.keras.models.load_model(saved_model_dir, compile=False)
    """
    model.load_weights(path_weight)
    return model


def _build_model_level_1():
    model = unetpp_level_1.create_model()
    # 推理端不 compile，不创建 optimizer，避免恢复时的 optimizer.* 告警
    return load_model_weights(model, PATH_WEIGHT_NETWORK_1)


def _build_model_level_2():
    model = unet_level_2.create_model()
    return load_model_weights(model, PATH_WEIGHT_NETWORK_2)


def load_segmentation_model():
    """
    供 FastAPI 启动时预热调用。
    返回字典：{"l1": model1, "l2": model2}
    """
    if _SEG_MODELS["l1"] is None:
        _SEG_MODELS["l1"] = _build_model_level_1()
    if _SEG_MODELS["l2"] is None:
        _SEG_MODELS["l2"] = _build_model_level_2()
    return _SEG_MODELS


# 模块导入时懒加载（也可不提前加载，让 FastAPI 在startup里调用）
try:
    load_segmentation_model()
except Exception as _e:
    # 权重路径不存在时，这里不要崩；等实际调用前再报错即可
    pass

def generate_crops(image_path, output_dir, window_size=228, stride=114,
                   resize_to=(912, 912)):   # ⭐ 新增：统一缩放
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[裁剪警告] 无法读取图像: {image_path}")
        return

    if resize_to is not None:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_AREA)

    h, w, _ = image.shape
    count = 0
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            crop = image[y:y + window_size, x:x + window_size]
            save_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_crop{count}.png"
            )
            cv2.imwrite(save_path, crop)
            count += 1
    print(f"[裁剪完成] 已生成 {count} 个裁剪图像，保存到: {output_dir}")
    return output_dir


def network_1_prediction(img_path, model_l1=None):
    """对单张图像进行一级网络预测"""
    if model_l1 is None:
        model_l1 = _SEG_MODELS["l1"]
        if model_l1 is None:
            raise RuntimeError("Level-1 模型尚未加载。请先调用 load_segmentation_model()。")

    # 加载并预处理图像（统一大小）
    img = load_img(img_path, grayscale=False, target_size=[IMG_HEIGHT, IMG_WIDTH])
    img = img_to_array(img).astype('float32') / 255.0
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)

    # 预测（静音）
    result = model_l1.predict(img, verbose=0)
    result = result.reshape(IMG_HEIGHT, IMG_WIDTH)
    result = (result > 0.5).astype(np.uint8)  # 二值化
    return result


def network_2_predict_single(crop_path, model_l2=None):
    """二级网络：单个 crop 的预测，返回二值掩码"""
    if model_l2 is None:
        model_l2 = _SEG_MODELS["l2"]
        if model_l2 is None:
            raise RuntimeError("Level-2 模型尚未加载。请先调用 load_segmentation_model()。")

    scale_height, scale_width = 128, 128

    # 读取并缩放
    img = load_img(crop_path, target_size=[scale_height, scale_width])
    img = img_to_array(img).astype('float32') / 255.0
    img = img.reshape(1, scale_height, scale_width, 3)

    # 预测
    pred = model_l2.predict(img, verbose=0).reshape(scale_height, scale_width)

    # 还原到 228×228
    pred_resized = cv2.resize(pred, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_NEAREST)

    # 二值化
    pred_resized = (pred_resized >= 0.05).astype(np.uint8)
    return pred_resized


def extract_crop_num(result):
    """提取包含燃烧区域的裁剪窗口（按行优先扫描）"""
    crop_num = 1
    save_crop_burned_num = []
    for row in np.arange(0, IMG_HEIGHT, CROP_HEIGHT):
        for col in np.arange(0, IMG_WIDTH, CROP_WIDTH):
            crop_window = result[row:row + CROP_HEIGHT, col:col + CROP_WIDTH]
            if crop_window.any():
                save_crop_burned_num.append(crop_num)
            crop_num += 1
    return save_crop_burned_num


def merge_mask_tiles_to_png(
    crop_dir,
    output_path,
    window_size=228,
    stride=114,
    resize_to=(912, 912),
    reference_image=None
):
    """
    将裁剪预测的掩码块目录拼回整图。
    参数:
        crop_dir: 掩码块所在文件夹
        output_path: 输出合并掩码的路径
        window_size, stride, resize_to: 与 generate_crops() 相同
        reference_image: 若指定，将输出缩放为该图的原始大小
    """
    files = sorted(
        [f for f in os.listdir(crop_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_crop")[-1])
    )

    if not files:
        print(f"[merge] 没找到掩码块: {crop_dir}")
        return None

    sample = cv2.imread(os.path.join(crop_dir, files[0]), cv2.IMREAD_UNCHANGED)
    h_tile, w_tile = sample.shape[:2]

    full_h, full_w = resize_to
    ny = (full_h - window_size) // stride + 1
    nx = (full_w - window_size) // stride + 1

    merged = np.zeros((full_h, full_w), dtype=np.float32)
    weight = np.zeros_like(merged)

    count = 0
    for j in range(ny):
        for i in range(nx):
            if count >= len(files):
                break
            path = os.path.join(crop_dir, files[count])
            tile = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            y, x = j * stride, i * stride
            merged[y:y+window_size, x:x+window_size] += tile
            weight[y:y+window_size, x:x+window_size] += 1.0
            count += 1

    weight[weight == 0] = 1.0
    merged /= weight
    merged = (merged > 0.5).astype(np.uint8) * 255

    # 若提供了参考图，则 resize 到原始图大小
    if reference_image and os.path.exists(reference_image):
        ref = cv2.imread(reference_image)
        ref_h, ref_w = ref.shape[:2]
        merged = cv2.resize(merged, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)

    # 将白色(255)变为红色
    mask_color = np.zeros((*merged.shape, 3), dtype=np.uint8)
    mask_color[merged == 255] = [0, 0, 255]  # BGR 红
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 如果你后面要按RGB读取统计“红色”，就把彩色图写盘：
    cv2.imwrite(output_path, mask_color)  # ← 改成保存彩色
    print(f"[merge] 已生成完整【彩色】掩码图: {output_path}")
    return output_path

# =================== Modified predict_mask ===================
def predict_mask(corrected_img_path, save_mask_dir, models: dict | None = None):
    """
    对矫正图像生成每个 crop 的掩码（RGB 红色区域为火灾）。
    models 可选：{"l1": model1, "l2": model2}
    """
    models = models or _SEG_MODELS  # 默认用模块级单例
    if models.get("l1") is None or models.get("l2") is None:
        # 尝试加载（如果之前没加载）
        load_segmentation_model()

    # （可选）一级网络粗预测 —— 目前仅用于调试/可视化，不做筛选
    try:
        _ = network_1_prediction(corrected_img_path, model_l1=models.get("l1"))
    except Exception as e:
        print(f"[警告] 一级网络预测失败（不影响二级）：{e}")

    # 生成 crop
    crop_dir = os.path.join(os.path.dirname(corrected_img_path), "crops")
    os.makedirs(crop_dir, exist_ok=True)
    generate_crops(corrected_img_path, crop_dir, window_size=CROP_HEIGHT, stride=CROP_HEIGHT // 2)

    # 二级网络逐块预测
    label_dir = os.path.join(save_mask_dir, "label_crops")
    os.makedirs(label_dir, exist_ok=True)

    crop_files = sorted([
        os.path.join(crop_dir, f)
        for f in os.listdir(crop_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])

    print(f"检测到 {len(crop_files)} 个裁剪块，开始生成掩码...")

    for i, crop_path in enumerate(crop_files):
        pred_mask = network_2_predict_single(crop_path, model_l2=models.get("l2"))

        mask_bin = (pred_mask.astype(np.uint8)) * 255  # 0 or 255

        mask_name = f"mask_{os.path.basename(crop_path)}"
        mask_path = os.path.join(label_dir, mask_name)
        cv2.imwrite(mask_path, mask_bin)

        if (i + 1) % 50 == 0 or i == len(crop_files) - 1:
            print(f"[{i + 1}/{len(crop_files)}] 保存掩码至: {mask_path}")

    print(f"✅ 所有掩码已保存至: {label_dir}")
    return label_dir


# -------------------------- 3. 红色面积计算模块（来自calculate_Area.py） --------------------------
import os
import numpy as np
from PIL import Image

def calculate_red_area_image(mask_path, area):
    """
    计算单张红色掩码图（merged.png）中红色区域的面积占比。
    参数:
        mask_path: merged.png 文件路径
        area: 实际地面总面积（平方米）
    返回:
        red_ratio: 红色占比 (0~1)
        red_area: 过火面积 (平方米)
    """
    if not os.path.exists(mask_path):
        print(f"[错误] 找不到掩码文件: {mask_path}")
        return 0, 0

    # 打开图像并转为 RGB
    img = Image.open(mask_path).convert('RGB')
    img_array = np.array(img)

    # 提取 R/G/B 通道
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # 红色像素判断（R=255, G=0, B=0）
    red_pixels = (r == 255) & (g == 0) & (b == 0)

    total_red = np.sum(red_pixels)
    total_pixels = img_array.shape[0] * img_array.shape[1]

    if total_pixels == 0:
        return 0, 0

    red_ratio = total_red / total_pixels
    red_area = red_ratio * area

    print(f"✅ 掩码统计完成：红色占比 {red_ratio:.4%}，过火面积约 {red_area:.2f} m²")
    return red_ratio, red_area


# -------------------------- 主流程：整合三个模块 --------------------------
if __name__ == "__main__":
    # 配置路径
    original_img_path = "/Users/xiaoyu/Downloads/forest-fire-damage-mapping-main/sample_data/sample_location_1_data/Orig/orig.JPG"  # 原始无人机图像路径
    mask_dir = "./sample_data/sample_location_1_data/Label"  # 掩码图保存目录

    area = 2000

    # 预测掩码（逐 crop）
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = predict_mask(original_img_path, mask_dir)

    merged_path = merge_mask_tiles_to_png(
        crop_dir="sample_data/sample_location_1_data/Label/label_crops",
        output_path="uploads/merged.png",
        window_size=228,
        stride=114,
        resize_to=(912, 912),
        reference_image="uploads/mission_1002.jpg"
    )

    # 计算红色像素数（所有掩码块的红像素累计）
    red_ratio, red_area = calculate_red_area_image("uploads/merged.png", area)

    print(f"🔥 估计燃烧面积：{red_area:.2f} m^2 （基于透视矫正与地面四边形标定）")
    print(f"（整幅图地面覆盖≈ {area} m^2；红色像素占比≈ {red_ratio:.2%}）")