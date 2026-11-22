import io
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import pillow_heif
import rawpy
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
from skimage.segmentation import find_boundaries, slic
from streamlit_image_coordinates import streamlit_image_coordinates

# HEIC/HEIFサポートを有効化
pillow_heif.register_heif_opener()

st.set_page_config(
    page_title="HSV + スーパーピクセル セグメンテーション",
    layout="wide",
)

DISPLAY_WIDTH = 750

# ------------------------------------------------------------
# 色相環カスタムコンポーネント (HTML5 Canvas ドラッグ対応)
# ------------------------------------------------------------
HUE_WHEEL_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: sans-serif;
    background: transparent;
    user-select: none;
  }
  canvas {
    cursor: grab;
    touch-action: none;
  }
  canvas.dragging {
    cursor: grabbing;
  }
  .info {
    margin-top: 4px;
    font-size: 13px;
    color: #555;
  }
</style>
</head>
<body>
  <canvas id="wheel" width="240" height="240"></canvas>
  <div class="info" id="info">H: 35 ～ 85 (0～179)</div>
<script>
  const canvas = document.getElementById("wheel");
  const ctx = canvas.getContext("2d");
  const infoEl = document.getElementById("info");

  const cx = 120, cy = 120;
  const rOuter = 104, rInner = 64;
  let hMin = 35;
  let hMax = 85;
  let lastSyncId = -1;
  let dragging = null; // "min" | "max" | "range" | null
  let dragStartAngle = 0;
  let startHMin = 35, startHMax = 85;

  function sendMessage(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }

  function hToAngle(h) {
    return (h / 180) * Math.PI * 2 - Math.PI / 2;
  }

  function angleToH(angle) {
    let deg = ((angle + Math.PI / 2) * 180 / Math.PI) % 360;
    if (deg < 0) deg += 360;
    return Math.round(deg / 2) % 180;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let deg = 0; deg < 360; deg++) {
      const a0 = (deg - 0.8) * Math.PI / 180 - Math.PI / 2;
      const a1 = (deg + 0.8) * Math.PI / 180 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, rOuter, a0, a1);
      ctx.arc(cx, cy, rInner, a1, a0, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${deg}, 100%, 50%)`;
      ctx.fill();
    }

    const aMin = hToAngle(hMin);
    const aMax = hToAngle(hMax);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, rInner - 4, aMin, aMax, false);
    ctx.closePath();
    ctx.fillStyle = "rgba(0, 160, 255, 0.22)";
    ctx.fill();

    function drawHandle(h, color) {
      const a = hToAngle(h);
      const x2 = cx + Math.cos(a) * (rOuter + 4);
      const y2 = cy + Math.sin(a) * (rOuter + 4);

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
      const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
      ctx.beginPath();
      ctx.arc(hx, hy, 8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }

    drawHandle(hMin, "#ffffff");
    drawHandle(hMax, "#ffeb3b");

    infoEl.textContent = `H (色相): ${hMin} ～ ${hMax}  (ドラッグで範囲調整)`;
  }

  function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function handleDist(h, px, py) {
    const a = hToAngle(h);
    const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
    const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
    return Math.hypot(px - hx, py - hy);
  }

  canvas.addEventListener("pointerdown", (e) => {
    const { x, y } = getPointerPos(e);
    const dMin = handleDist(hMin, x, y);
    const dMax = handleDist(hMax, x, y);

    if (dMin < 18 && dMin <= dMax) {
      dragging = "min";
    } else if (dMax < 18) {
      dragging = "max";
    } else {
      const dist = Math.hypot(x - cx, y - cy);
      if (dist <= rOuter + 10) {
        dragging = "range";
        dragStartAngle = Math.atan2(y - cy, x - cx);
        startHMin = hMin;
        startHMax = hMax;
      }
    }
    if (dragging) {
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add("dragging");
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const { x, y } = getPointerPos(e);
    const angle = Math.atan2(y - cy, x - cx);

    if (dragging === "min") {
      hMin = angleToH(angle);
    } else if (dragging === "max") {
      hMax = angleToH(angle);
    } else if (dragging === "range") {
      const diffRad = angle - dragStartAngle;
      const diffH = Math.round((diffRad / (Math.PI * 2)) * 180);
      hMin = ((startHMin + diffH) % 180 + 180) % 180;
      hMax = ((startHMax + diffH) % 180 + 180) % 180;
    }
    draw();
  });

  function endDrag() {
    if (!dragging) return;
    dragging = null;
    canvas.classList.remove("dragging");
    sendMessage("streamlit:setComponentValue", {
      value: { h_min: hMin, h_max: hMax },
      dataType: "json",
    });
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "streamlit:render") {
      const args = event.data.args || {};
      // プリセット変更などPython側から明示的にsync_idが更新されたときのみ値を同期する
      if (typeof args.sync_id === "number" && args.sync_id !== lastSyncId) {
        lastSyncId = args.sync_id;
        if (typeof args.h_min === "number") hMin = args.h_min;
        if (typeof args.h_max === "number") hMax = args.h_max;
        draw();
      }
      sendMessage("streamlit:setFrameHeight", { height: 270 });
    }
  });

  draw();
  sendMessage("streamlit:componentReady", { apiVersion: 1 });
  sendMessage("streamlit:setFrameHeight", { height: 270 });
</script>
</body>
</html>
"""


@st.cache_resource
def get_hue_wheel_component():
    comp_dir = os.path.join(tempfile.gettempdir(), "st_hue_wheel_component_v2")
    os.makedirs(comp_dir, exist_ok=True)
    index_path = os.path.join(comp_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(HUE_WHEEL_HTML)
    return components.declare_component("hue_wheel_v2", path=comp_dir)


hue_wheel_component = get_hue_wheel_component()


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
# 色プリセット & マスク表示色定義
# ------------------------------------------------------------
COLOR_PRESETS = {
    "緑": ((35, 85), (40, 255), (40, 255)),
    "赤": ((170, 10), (70, 255), (50, 255)),
    "青": ((95, 130), (50, 255), (50, 255)),
    "黄": ((18, 35), (70, 255), (70, 255)),
    "白": ((0, 179), (0, 50), (200, 255)),
    "黒": ((0, 179), (0, 255), (0, 50)),
}

MASK_DISPLAY_COLORS = {
    "水色": (0, 200, 255),
    "赤": (255, 60, 0),
    "緑": (0, 220, 80),
    "黄": (255, 220, 0),
    "紫": (180, 60, 255),
}


# ------------------------------------------------------------
# セッション状態初期化
# ------------------------------------------------------------
def init_state():
    defaults = {
        "file_key": None,
        "work_mask": None,
        "last_click": None,
        "last_preset": "緑",
        "h_min": 35,
        "h_max": 85,
        "s_range": (40, 255),
        "v_range": (40, 255),
        "wheel_sync_id": 0,
        "last_wheel_val": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ------------------------------------------------------------
# メインUI
# ------------------------------------------------------------
st.title("HSV + スーパーピクセル セグメンテーションアプリ")
st.caption("タブ①でHSV値により大雑把にマスクし、タブ②でスーパーピクセル単位で修正、ダウンロードタブから保存できます。")

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

tab1, tab2, tab3 = st.tabs([
    "① HSVでマスク作成",
    "②スーパーピクセルで修正",
    "ダウンロード",
])

# ------------------------------------------------------------
# タブ①: HSV値に従って大雑把に分ける
# ------------------------------------------------------------
with tab1:
    col1_ctrl, col1_view = st.columns([1, 2])

    with col1_ctrl:
        st.subheader("設定")
        preset = st.selectbox(
            "色プリセット",
            ["緑", "赤", "青", "黄", "白", "黒", "カスタム"],
        )

        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            if preset in COLOR_PRESETS:
                (h0, h1), s_def, v_def = COLOR_PRESETS[preset]
                st.session_state.h_min = h0
                st.session_state.h_max = h1
                st.session_state.s_range = s_def
                st.session_state.v_range = v_def
                st.session_state.wheel_sync_id += 1

        mask_color_name = st.selectbox(
            "マスクの色",
            list(MASK_DISPLAY_COLORS.keys()),
        )
        selected_mask_color = MASK_DISPLAY_COLORS[mask_color_name]

        st.markdown("**H (色相) 範囲**（色相環のハンドルまたは範囲をドラッグ）")
        wheel_val = hue_wheel_component(
            h_min=st.session_state.h_min,
            h_max=st.session_state.h_max,
            sync_id=st.session_state.wheel_sync_id,
            key="hue_wheel_widget",
        )
        if (
            isinstance(wheel_val, dict)
            and "h_min" in wheel_val
            and "h_max" in wheel_val
            and wheel_val != st.session_state.last_wheel_val
        ):
            st.session_state.last_wheel_val = wheel_val
            st.session_state.h_min = int(wheel_val["h_min"])
            st.session_state.h_max = int(wheel_val["h_max"])

        s_min, s_max = st.slider("S (彩度) 範囲", 0, 255, key="s_range")
        v_min, v_max = st.slider("V (明度) 範囲", 0, 255, key="v_range")

        hsv_mask = create_hsv_mask(
            img_rgb,
            st.session_state.h_min,
            st.session_state.h_max,
            s_min,
            s_max,
            v_min,
            v_max,
        )

        # 新しい画像が読み込まれたら初期マスクをセット
        if st.session_state.file_key != file_key:
            st.session_state.file_key = file_key
            st.session_state.work_mask = hsv_mask.copy()
            st.session_state.last_click = None

    with col1_view:
        st.subheader("プレビュー")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            hsv_preview = render_overlay(
                img_rgb,
                hsv_mask,
                None,
                mask_color=selected_mask_color,
                mask_alpha=0.5,
                show_boundaries=False,
            )
            st.image(hsv_preview, caption=f"元画像 + HSVマスク（{mask_color_name}）", use_container_width=True)
        with prev_col2:
            st.image((hsv_mask.astype(np.uint8)) * 255, caption="HSVマスク（白黒）", use_container_width=True)

        if st.button("このHSVマスクを②に反映", type="primary", use_container_width=True):
            st.session_state.work_mask = hsv_mask.copy()
            st.success("HSVマスクを「②スーパーピクセルで修正」に反映しました。")

# ------------------------------------------------------------
# タブ②: スーパーピクセルで修正
# ------------------------------------------------------------
with tab2:
    col2_ctrl, col2_main, col2_mask = st.columns([1, 2, 1])

    with col2_ctrl:
        st.subheader("設定")
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
        brush_radius = st.slider(
            "まとめてピクセルを選択",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="0にするとクリックした1つのスーパーピクセルのみを反転します。",
        )

        with st.spinner("スーパーピクセルを計算中..."):
            segments, boundaries = compute_superpixels(img_rgb, n_segments, compactness)

    with col2_main:
        st.subheader("編集画面")
        overlay_img = render_overlay(
            img_rgb,
            st.session_state.work_mask,
            boundaries,
            mask_color=selected_mask_color,
            show_boundaries=show_boundaries,
        )
        disp_img, scale = resize_for_display(overlay_img, DISPLAY_WIDTH)

        coords = streamlit_image_coordinates(
            Image.fromarray(disp_img),
            key="sp_canvas",
        )

        if coords is not None:
            click_sig = (coords["x"], coords["y"], brush_radius, n_segments)
            if st.session_state.last_click != click_sig:
                st.session_state.last_click = click_sig

                orig_x = int(np.clip(round(coords["x"] / scale), 0, img_w - 1))
                orig_y = int(np.clip(round(coords["y"] / scale), 0, img_h - 1))

                if brush_radius == 0:
                    target_label = segments[orig_y, orig_x]
                    target_region = segments == target_label
                else:
                    yy, xx = np.ogrid[:img_h, :img_w]
                    circle = (xx - orig_x) ** 2 + (yy - orig_y) ** 2 <= brush_radius ** 2
                    target_labels = np.unique(segments[circle])
                    target_region = np.isin(segments, target_labels)

                center_val = st.session_state.work_mask[orig_y, orig_x]
                st.session_state.work_mask[target_region] = not center_val

                st.rerun()

    with col2_mask:
        st.subheader("最終マスク画像（白黒）")
        binary_mask_preview = (st.session_state.work_mask.astype(np.uint8)) * 255
        st.image(binary_mask_preview, use_container_width=True)

# ------------------------------------------------------------
# タブ③: ダウンロード
# ------------------------------------------------------------
with tab3:
    st.subheader("ダウンロード")
    binary_mask_u8 = (st.session_state.work_mask.astype(np.uint8)) * 255
    final_overlay = render_overlay(
        img_rgb,
        st.session_state.work_mask,
        None,
        mask_color=selected_mask_color,
        show_boundaries=False,
    )

    mask_ratio = float(np.mean(st.session_state.work_mask) * 100.0)
    st.metric("マスク面積率", f"{mask_ratio:.2f}%")

    base_name = os.path.splitext(uploaded_file.name)[0]
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.image(img_rgb, caption="元画像", use_container_width=True)
        st.download_button(
            label="元画像をダウンロード (PNG)",
            data=encode_png(img_rgb),
            file_name=f"{base_name}_original.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_col2:
        st.image(binary_mask_u8, caption="最終マスク画像（白黒）", use_container_width=True)
        st.download_button(
            label="マスク画像をダウンロード (PNG)",
            data=encode_png(binary_mask_u8),
            file_name=f"{base_name}_mask.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )

    with dl_col3:
        st.image(final_overlay, caption="最終オーバーレイ画像", use_container_width=True)
        st.download_button(
            label="オーバーレイ画像をダウンロード (PNG)",
            data=encode_png(final_overlay),
            file_name=f"{base_name}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )
