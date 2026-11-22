import io
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import pillow_heif
import rawpy
import streamlit as st
from PIL import Image, ImageOps
from skimage.segmentation import find_boundaries, slic
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC/HEIFサポートを有効化
pillow_heif.register_heif_opener()

st.set_page_config(
    page_title="HSV + スーパーピクセル セグメンテーション",
    layout="wide",
)


# ------------------------------------------------------------
# 画像読み込みユーティリティ (.jpg / .heic / .cr2 対応)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_image_rgb(file_bytes: bytes, filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".cr2":
        with tempfile.NamedTemporaryFile(suffix=".cr2", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rawpy.imread(tmp_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            return rgb
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # .jpg, .jpeg, .png, .heic, .heif など
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ------------------------------------------------------------
# スーパーピクセル計算 (キャッシュ付き)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_superpixels(
    img_rgb: np.ndarray,
    n_segments: int,
    compactness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )
    boundaries = find_boundaries(segments, mode="inner")
    return segments.astype(np.int32), boundaries


# ------------------------------------------------------------
# HSVマスク生成
# ------------------------------------------------------------
def create_hsv_mask(
    img_rgb: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if h_min <= h_max:
        h_mask = (h >= h_min) & (h <= h_max)
    else:
        # 赤色など色相が0/179をまたぐ場合
        h_mask = (h >= h_min) | (h <= h_max)

    sv_mask = (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)
    return h_mask & sv_mask


def snap_mask_to_superpixels(
    mask: np.ndarray,
    segments: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """各スーパーピクセル内のマスク割合がthreshold以上ならそのスーパーピクセル全体を塗りつぶす"""
    flat_seg = segments.ravel()
    flat_mask = mask.ravel().astype(np.float32)

    total_counts = np.bincount(flat_seg)
    mask_counts = np.bincount(flat_seg, weights=flat_mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(total_counts > 0, mask_counts / total_counts, 0.0)

    active_labels = ratios >= threshold
    return active_labels[segments]


# ------------------------------------------------------------
# 描画・オーバーレイ生成
# ------------------------------------------------------------
def render_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    boundaries: Optional[np.ndarray],
    mask_color: Tuple[int, int, int] = (255, 60, 0),
    mask_alpha: float = 0.45,
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
    show_boundaries: bool = True,
) -> np.ndarray:
    canvas = img_rgb.astype(np.float32).copy()

    if np.any(mask):
        color_arr = np.array(mask_color, dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - mask_alpha) + color_arr * mask_alpha

    out = np.clip(canvas, 0, 255).astype(np.uint8)

    if show_boundaries and boundaries is not None:
        out[boundaries] = boundary_color

    return out


def resize_for_display(img: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(
        img,
        (max_width, new_h),
        interpolation=cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA,
    )
    return resized, scale


def encode_png(img_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------
# セッション状態初期化
# ------------------------------------------------------------
def init_state():
    defaults = {
        "file_key": None,
        "work_mask": None,
        "history": [],
        "last_click": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ------------------------------------------------------------
# メインUI
# ------------------------------------------------------------
st.title("HSV + スーパーピクセル セグメンテーションアプリ")
st.caption("タブ①でHSV値により大雑把にマスクし、タブ②でスーパーピクセル単位の修正とダウンロードを行います。")

uploaded_file = st.file_uploader(
    "画像をアップロード (.jpg / .jpeg / .png / .heic / .cr2)",
    type=["jpg", "jpeg", "png", "heic", "heif", "cr2"],
)

if uploaded_file is None:
    st.info("上のアップローダーから画像ファイルを選択してください。")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_key = f"{uploaded_file.name}_{len(file_bytes)}"

with st.spinner("画像を読み込んでいます..."):
    img_rgb = load_image_rgb(file_bytes, uploaded_file.name)

img_h, img_w = img_rgb.shape[:2]

tab1, tab2 = st.tabs([
    "① HSVでマスク作成",
    "② スーパーピクセルで修正・③ ダウンロード",
])

# ------------------------------------------------------------
# タブ①: HSV値に従って大雑把に分ける
# ------------------------------------------------------------
with tab1:
    col1_ctrl, col1_view = st.columns([1, 2])

    with col1_ctrl:
        st.subheader("① HSVマスク設定")
        preset = st.selectbox(
            "色プリセット",
            ["カスタム", "緑（葉・植物）", "赤（色相またぎ）", "青（空・水）"],
        )

        if preset == "緑（葉・植物）":
            def_h, def_s, def_v = (35, 85), (40, 255), (40, 255)
        elif preset == "赤（色相またぎ）":
            def_h, def_s, def_v = (170, 10), (70, 255), (50, 255)
        elif preset == "青（空・水）":
            def_h, def_s, def_v = (95, 130), (50, 255), (50, 255)
        else:
            def_h, def_s, def_v = (35, 85), (40, 255), (40, 255)

        h_min = st.slider("H (色相) 最小", 0, 179, def_h[0])
        h_max = st.slider("H (色相) 最大", 0, 179, def_h[1])
        s_min, s_max = st.slider("S (彩度) 範囲", 0, 255, def_s)
        v_min, v_max = st.slider("V (明度) 範囲", 0, 255, def_v)

        hsv_mask = create_hsv_mask(img_rgb, h_min, h_max, s_min, s_max, v_min, v_max)
        hsv_ratio = float(np.mean(hsv_mask) * 100.0)
        st.metric("HSV抽出面積率", f"{hsv_ratio:.2f}%")

        # 新しい画像が読み込まれたら初期マスクをセット
        if st.session_state.file_key != file_key:
            st.session_state.file_key = file_key
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.history = []
            st.session_state.last_click = None

        if st.button("このHSVマスクを②の作業用マスクに反映", type="primary", use_container_width=True):
            if st.session_state.work_mask is not None:
                st.session_state.history.append(st.session_state.work_mask.copy())
            st.session_state.work_mask = hsv_mask.copy()
            st.success("HSVマスクを作業用マスクに反映しました。「② スーパーピクセルで修正」タブへ進んでください。")

    with col1_view:
        st.subheader("HSVマスクプレビュー")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            hsv_preview = render_overlay(
                img_rgb,
                hsv_mask,
                None,
                mask_color=(0, 200, 255),
                mask_alpha=0.5,
                show_boundaries=False,
            )
            st.image(hsv_preview, caption="元画像 + HSVマスク（水色）", use_container_width=True)
        with prev_col2:
            st.image((hsv_mask.astype(np.uint8)) * 255, caption="HSVマスク（白黒）", use_container_width=True)

# ------------------------------------------------------------
# タブ②: スーパーピクセルで修正 & ダウンロード
# ------------------------------------------------------------
with tab2:
    col2_ctrl, col2_main, col2_out = st.columns([1, 2, 1])

    with col2_ctrl:
        st.subheader("②・④ スーパーピクセル設定")
        n_segments = st.slider(
            "スーパーピクセルの粒度（分割数）",
            min_value=50,
            max_value=1500,
            value=250,
            step=10,
            help="数値が大きいほどスーパーピクセルが細かくなります。",
        )
        compactness = st.slider(
            "コンパクトネス（形状の均一さ）",
            min_value=1.0,
            max_value=40.0,
            value=10.0,
            step=1.0,
        )
        show_boundaries = st.checkbox("スーパーピクセル境界線を表示", value=True)
        snap_threshold = st.slider("HSVスナップ閾値（領域内の割合）", 0.1, 0.9, 0.5, 0.05)

        with st.spinner("スーパーピクセルを計算中..."):
            segments, boundaries = compute_superpixels(img_rgb, n_segments, compactness)

        st.divider()
        st.subheader("編集ツール")
        edit_mode = st.radio(
            "クリック時の動作",
            ["追加（塗る）", "削除（消す）", "反転（トグル）"],
        )
        brush_radius = st.slider(
            "ブラシ半径 (px・元画像基準)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="0にするとクリックした1つのスーパーピクセルのみを変更します。",
        )
        display_width = st.slider("表示サイズ (px)", 400, 1200, 750, 50)

        st.divider()
        if st.button("HSVマスクをスーパーピクセルにスナップ", use_container_width=True):
            st.session_state.history.append(st.session_state.work_mask.copy())
            st.session_state.work_mask = snap_mask_to_superpixels(hsv_mask, segments, snap_threshold)

        if st.button("HSVマスクで初期化（ピクセル）", use_container_width=True):
            st.session_state.history.append(st.session_state.work_mask.copy())
            st.session_state.work_mask = hsv_mask.copy()

        if st.button("1つ前に戻る (Undo)", use_container_width=True, disabled=len(st.session_state.history) == 0):
            st.session_state.work_mask = st.session_state.history.pop()

        if st.button("マスクを全消去", use_container_width=True):
            st.session_state.history.append(st.session_state.work_mask.copy())
            st.session_state.work_mask = np.zeros((img_h, img_w), dtype=bool)

    with col2_main:
        st.subheader("スーパーピクセル編集キャンバス（クリックで追加/削除）")
        overlay_img = render_overlay(
            img_rgb,
            st.session_state.work_mask,
            boundaries,
            show_boundaries=show_boundaries,
        )
        disp_img, scale = resize_for_display(overlay_img, display_width)

        coords = streamlit_image_coordinates(
            Image.fromarray(disp_img),
            key="sp_canvas",
        )

        if coords is not None:
            click_sig = (coords["x"], coords["y"], edit_mode, brush_radius, n_segments)
            if st.session_state.last_click != click_sig:
                st.session_state.last_click = click_sig

                orig_x = int(np.clip(round(coords["x"] / scale), 0, img_w - 1))
                orig_y = int(np.clip(round(coords["y"] / scale), 0, img_h - 1))

                st.session_state.history.append(st.session_state.work_mask.copy())
                if len(st.session_state.history) > 30:
                    st.session_state.history.pop(0)

                if brush_radius == 0:
                    target_label = segments[orig_y, orig_x]
                    target_region = segments == target_label
                else:
                    yy, xx = np.ogrid[:img_h, :img_w]
                    circle = (xx - orig_x) ** 2 + (yy - orig_y) ** 2 <= brush_radius ** 2
                    target_labels = np.unique(segments[circle])
                    target_region = np.isin(segments, target_labels)

                if edit_mode == "追加（塗る）":
                    st.session_state.work_mask[target_region] = True
                elif edit_mode == "削除（消す）":
                    st.session_state.work_mask[target_region] = False
                else:
                    center_val = st.session_state.work_mask[orig_y, orig_x]
                    st.session_state.work_mask[target_region] = not center_val

                st.rerun()

    with col2_out:
        st.subheader("③ マスク出力・ダウンロード")
        binary_mask_u8 = (st.session_state.work_mask.astype(np.uint8)) * 255
        st.image(binary_mask_u8, caption="最終マスク画像（白黒）", use_container_width=True)

        mask_ratio = float(np.mean(st.session_state.work_mask) * 100.0)
        st.metric("マスク面積率", f"{mask_ratio:.2f}%")

        base_name = os.path.splitext(uploaded_file.name)[0]
        st.download_button(
            label="マスク画像をダウンロード (PNG)",
            data=encode_png(binary_mask_u8),
            file_name=f"{base_name}_mask.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

        final_overlay = render_overlay(img_rgb, st.session_state.work_mask, None, show_boundaries=False)
        st.download_button(
            label="オーバーレイ画像をダウンロード (PNG)",
            data=encode_png(final_overlay),
            file_name=f"{base_name}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )
