import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import base64
from datetime import datetime
import signal
import threading
import time
import sys
import tempfile
from ultralytics import YOLO

import nibabel as nib
import nrrd

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

def make_model_input(ct_array, roi_array, side=64):
    coords = np.where(roi_array > 0)
    if len(coords[0]) == 0:
        return draw

    mi_y = np.min(coords[0])
    mi_x = np.min(coords[1])
    mi_z = np.min(coords[2])

    # 在 ROI 中心附近取 64×64，並連續取 z ~ z+8 層
    pa_s = side // 2  # half of 64
    start_x = mi_x - pa_s
    end_x   = mi_x + pa_s
    start_y = mi_y - pa_s
    end_y   = mi_y + pa_s
    start_z = mi_z
    end_z   = mi_z + 8

    # 邊界檢查
    if start_x < 0:
        end_x += (0 - start_x)
        start_x = 0
    if end_x > ct_array.shape[1]:
        start_x -= (end_x - ct_array.shape[1])
        end_x = ct_array.shape[1]

    if start_y < 0:
        end_y += (0 - start_y)
        start_y = 0
    if end_y > ct_array.shape[0]:
        start_y -= (end_y - ct_array.shape[0])
        end_y = ct_array.shape[0]

    if end_z > ct_array.shape[2]:
        start_z -= (end_z - ct_array.shape[2])
        end_z = ct_array.shape[2]
    if start_z < 0:
        end_z += (0 - start_z)
        start_z = 0

    # 依序把 9 層切片擺到 draw
    # slice_0 放在左上, slice_1 放在中上, slice_2 放在右上, ...
    # 同 make_yoloimg 的邏輯

    # for i in range(9): => z + i
    # row = i // 3, col = i % 3
    # draw[row*side : (row+1)*side, col*side : (col+1)*side] = 這一層

    for i in range(9):
        z_idx = start_z + i
        if z_idx >= end_z:
            break

        slice_2d = ct_array[start_y:end_y, start_x:end_x, z_idx]

        row = i // 3
        col = i % 3
        # 先檢查當前 slice_2d 的 shape
        h, w = slice_2d.shape
        # 如果實際拿到的 patch < 64, 就把它置中或直接貼左上?
        # 這裡簡單做：若比64小，就放左上
        # 若比64大, 表示切片box比64大, (理論上不太會?), 也就截斷了
        patch = np.zeros((side, side), dtype=np.uint8)

        hh = min(h, side)
        ww = min(w, side)
        patch[0:hh, 0:ww] = slice_2d[0:hh, 0:ww]

        draw[row*side : (row+1)*side, col*side : (col+1)*side] = patch

    return draw

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

lab_logo = get_image_base64('./logo_lab.png')
hos_logo = get_image_base64('./logo_hos.png')

st.set_page_config(page_title='肺結節CT診斷系統', layout='wide')

# 樣式設定
st.markdown(
    '''
    <style>
    .header-bar {
        background-color: #E0F0FF;
        padding: 0;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        left: 0;
        right: 0;
        height: 130px;
        z-index: 9999;
        border-bottom: 1px solid #ddd;
        width: 100%;
    }
    .header-content {
        display: flex;
        align-items: center;
    }
    .header-bar h1 {
        font-size: 32px;
        margin: 0;
        font-style: italic;
        text-align: center;
        flex-grow: 1;
        font-family: 'Arial', sans-serif;
        color: #0047AB;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #888;
    }
    .prediction-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 7px;
        text-align: center;
        font-size: 24px;
        background-color: #F8F8F8;
        transition: all 0.3s;
    }
    .highlight-box {
        border-radius: 10px;
        padding: 10px;
        margin: 7px;
        text-align: center;    
        background-color: #C1FFC1;
        border: 2px solid #4CAF50;
        font-size: 28px;
        transform: scale(1.03);
        font-weight: bold;
        transition: all 0.3s;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    @media (max-width: 1200px) {
        .grid-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 800px) {
        .grid-container {
            grid-template-columns: repeat(1, 1fr);
        }
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# 頁首標題區
st.markdown(
    f'''
    <div class="header-bar">
        <div class="header-content">
            <img src="data:image/png;base64,{lab_logo}" alt="Lab Logo" style="height: 125px; margin-left: 20px;">
        </div>
        <h1>Lung Nodule Diagnosis Tool</h1>
        <div class="header-content">
            <img src="data:image/png;base64,{hos_logo}" alt="Hospital Logo" style="height: 125px; margin-right: 20px;">
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('<div style="margin-top: 110px;"></div>', unsafe_allow_html=True)

# ------------------ Windowing Function ------------------
def apply_lung_window(ct_data, window_center=-600, window_width=1500):
    """
    對整個 3D CT array 做一次性的 Lung Window 處理
    回傳值為 0~255 的 uint8 3D array
    """
    low = window_center - (window_width / 2.0)  # -600 - 750 = -1350
    high = window_center + (window_width / 2.0) # -600 + 750 = 150

    clamped = np.clip(ct_data, low, high)
    # 線性縮放到 0~255
    out = ( (clamped - low) / (high - low) ) * 255.0
    out = np.round(out).astype(np.uint8)
    return out

# 側邊欄：上傳 CT 與 Contour
st.sidebar.title("上傳 NIFTI/NRRD 檔案")
ct_file = st.sidebar.file_uploader("請選擇 CT 三維檔 (lung window)", type=["nii", "nii.gz", "nrrd"])
mask_file = st.sidebar.file_uploader("請選擇 Contour 三維檔", type=["nii", "nii.gz", "nrrd"])
model = YOLO("./best_nodule.pt")
col1, col2 = st.columns([1.7, 1.3])

CT_DATA = None
MASK_DATA = None

with col1:
    if ct_file and mask_file:
        # 讀取 CT
        with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as tmp_ct:
            tmp_ct.write(ct_file.read())
            ct_path = tmp_ct.name
        CT_DATA, ct_header = nrrd.read(ct_path)
        os.remove(ct_path)

        # 讀取 Mask
        with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as tmp_mask:
            tmp_mask.write(mask_file.read())
            mask_path = tmp_mask.name
        MASK_DATA, mask_header = nrrd.read(mask_path)
        os.remove(mask_path)

        # 檢查維度
        if CT_DATA.shape != MASK_DATA.shape:
            st.error("CT 和 Contour 的維度不一致，請檢查！")
        else:
            st.markdown("<h3 style='text-align: center;'>CT + Contour 顯示</h3>", unsafe_allow_html=True)

            # ------ 找出 ROI 的 z 索引 ------
            roi_zs = np.where(MASK_DATA > 0)[2]
            roi_zs = np.unique(roi_zs)
            z_max = CT_DATA.shape[2] - 1
            default_z = int(roi_zs[0]) if len(roi_zs) > 0 else 0

            # ------ 初始化 session_state ------
            if "z_index" not in st.session_state:
                st.session_state["z_index"] = default_z
            if "show_contour" not in st.session_state:
                st.session_state["show_contour"] = True
            if "rotation_count" not in st.session_state:
                st.session_state["rotation_count"] = 0
            # **存放已經windowing好的資料**, 避免重複計算
            if "CT_WINDOWED" not in st.session_state:
                # 只做一次 lung window
                st.session_state["CT_WINDOWED"] = apply_lung_window(CT_DATA, -600, 1500)

            # ========== 按鈕列 ==========
            col_dec, col_num, col_inc = st.columns([1, 3, 1])
            col_toggle, col_rotate = st.columns([2,2])

            # 上一層
            with col_dec:
                if st.button("上一層"):
                    st.session_state["z_index"] = max(st.session_state["z_index"] - 1, 0)

            # 輸入切片編號
            with col_num:
                z_value = st.number_input(
                    "Z 軸切片",
                    min_value=0,
                    max_value=z_max,
                    value=st.session_state["z_index"],
                    step=1
                )
                st.session_state["z_index"] = z_value

            # 下一層
            with col_inc:
                if st.button("下一層"):
                    st.session_state["z_index"] = min(st.session_state["z_index"] + 1, z_max)

            # 顯示 / 隱藏 Contour
            with col_toggle:
                if st.button("顯示 / 隱藏 Contour"):
                    st.session_state["show_contour"] = not st.session_state["show_contour"]

            # 旋轉90度
            with col_rotate:
                if st.button("旋轉90度"):
                    st.session_state["rotation_count"] = (st.session_state["rotation_count"] + 1) % 4

            # ====== 顯示影像 ======
            z_index = st.session_state["z_index"]
            # 取出windowing後的該z切片
            ct_slice = st.session_state["CT_WINDOWED"][:, :, z_index]
            mask_slice = MASK_DATA[:, :, z_index]

            # 若有旋轉
            if st.session_state["rotation_count"] != 0:
                ct_slice = np.rot90(ct_slice, k=-st.session_state["rotation_count"])
                mask_slice = np.rot90(mask_slice, k=-st.session_state["rotation_count"])

            # 轉成 RGBA
            base_img = Image.fromarray(ct_slice, mode="L").convert("RGBA")

            # contour 疊加
            overlay_rgba = np.zeros((ct_slice.shape[0], ct_slice.shape[1], 4), dtype=np.uint8)
            if st.session_state["show_contour"]:
                overlay_rgba[mask_slice > 0] = [255, 0, 0, int(0.5 * 255)]
            overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")

            blended = Image.alpha_composite(base_img, overlay_img)
            st.image(blended, use_container_width=True)

    else:
        st.info("請同時上傳 CT 與 Contour 的檔案，以查看影像。")

with col2:
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            font-size: 24px; 
            padding: 10px 20px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("開始預測"):
        # === 這裡示範如何把原本 3D array (CT_DATA) + ROI (MASK_DATA) 轉成 2D 影像 ===
        model_input_2d = make_model_input(CT_DATA, MASK_DATA)

        # 接著就可以把這個 model_input_2d 傳給您實際的模型
        # 這裡仍以亂數示範
        pre = model.predict(model_input_2d, verbose=False)
        pre = pre[0].probs.data.cpu().numpy()
        p_benign = pre[0]
        p_malignant = pre[1]
        st.session_state["prediction_dict"] = {
            "良性結節(Benign)": p_benign,
            "惡性結節(Malignant)": p_malignant
        }

    if "prediction_dict" in st.session_state:
        pred_dict = st.session_state["prediction_dict"]
        st.markdown(
            '''
            <div class="result-title" style="text-align: center; font-size: 30px; font-weight: bold;">
                預測結果
            </div>
            ''', 
            unsafe_allow_html=True
        )
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)

        max_prob = max(pred_dict.values())
        for label, prob in pred_dict.items():
            box_class = "highlight-box" if prob == max_prob else "prediction-box"
            st.markdown(f'<div class="{box_class}">{label}: {prob*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ 底部版權 ------------------
st.markdown(
    '''
    <div class="footer">
        © 2025 CMU Artificial Intelligence & Bioimaging Lab, Kaohsiung Veterans General Hospital. All rights reserved.<br>
        Developed by: Chen-Hao Peng & Da-Chuan Cheng & Kuan-Jung Chen & Ke-An Hong
    </div>
    ''', 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-top: 10px;">
        <img src="data:image/png;base64,{lab_logo}" alt="Lab Logo" style="max-width: 150px;">
    </div>
    """,
    unsafe_allow_html=True
)
