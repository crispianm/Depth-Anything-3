import cv2
import torch
import numpy as np
from depth_anything_3.api import DepthAnything3

# 1. Load Depth Anything 3 Metric Large model
model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
model.eval()

# --- Focal map settings ---
K_VALUE = 50.0   # blur strength; raise for more selective focus
MAX_COC = 100.0   # disparity range for normalisation
CAMERA_FOCAL_PX = None  # set this if known for better absolute metric scale
ASSUMED_FOV_DEG = 60.0   # fallback when CAMERA_FOCAL_PX is unknown
PANEL_WIDTH = 540

# --- Streaming settings (DA3-Streaming style: chunk + overlap) ---
CHUNK_SIZE = 5
OVERLAP = 4
PROCESS_RES = 518
REF_VIEW_STRATEGY = "middle"

if OVERLAP >= CHUNK_SIZE:
    raise ValueError("OVERLAP must be smaller than CHUNK_SIZE")

_click = [None, None]
_log_click_values = False
_first_chunk = True
_rgb_chunk = []
_bgr_chunk = []
_latest_depth_metric = None
_latest_frame_bgr = None

def _on_mouse(event, x, y, flags, param):
    global _log_click_values
    if event == cv2.EVENT_LBUTTONDOWN:
        # Modulo panel width so clicks on all 3 panels map to the image domain
        _click[0] = x % PANEL_WIDTH
        _click[1] = y
        _log_click_values = True

cap = cv2.VideoCapture(-1)
# OPTIMIZATION: Force a fast hardware-friendly decode format if your webcam supports it
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow("Focus Map")
cv2.setMouseCallback("Focus Map", _on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame early
    frame_small = cv2.resize(frame, (PANEL_WIDTH, 400))
    h, w = frame_small.shape[:2]
    frame = frame_small

    # Prepare RGB input for DA3
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Streaming buffers (chunked inference with overlap)
    _rgb_chunk.append(img_rgb)
    _bgr_chunk.append(frame.copy())

    if len(_rgb_chunk) >= CHUNK_SIZE:
        prediction = model.inference(
            _rgb_chunk,
            process_res=PROCESS_RES,
            process_res_method="upper_bound_resize",
            ref_view_strategy=REF_VIEW_STRATEGY,
        )

        focal_px = CAMERA_FOCAL_PX
        if focal_px is None or focal_px <= 0:
            focal_px = (w * 0.5) / np.tan(np.deg2rad(ASSUMED_FOV_DEG * 0.5))

        depth_canonical_chunk = prediction.depth
        depth_metric_chunk = (focal_px * depth_canonical_chunk) / 300.0

        if _first_chunk:
            emit_start = 0
            emit_end = CHUNK_SIZE - OVERLAP
            _first_chunk = False
        else:
            emit_start = OVERLAP
            emit_end = CHUNK_SIZE

        if emit_end > emit_start:
            latest_idx = emit_end - 1
            _latest_depth_metric = depth_metric_chunk[latest_idx]
            _latest_frame_bgr = _bgr_chunk[latest_idx].copy()

        # Keep only overlap tail for next chunk
        _rgb_chunk = _rgb_chunk[-OVERLAP:]
        _bgr_chunk = _bgr_chunk[-OVERLAP:]

    if _latest_depth_metric is None or _latest_frame_bgr is None:
        waiting = frame.copy()
        cv2.putText(
            waiting,
            f"Buffering for streaming... ({len(_rgb_chunk)}/{CHUNK_SIZE})",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Focus Map", waiting)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame = _latest_frame_bgr
    depth_metric = _latest_depth_metric

    # Align RGB frame to the depth map resolution used by DA3
    depth_h, depth_w = depth_metric.shape
    if frame.shape[0] != depth_h or frame.shape[1] != depth_w:
        frame = cv2.resize(frame, (depth_w, depth_h))
    h, w = frame.shape[:2]

    # Keep focus-map math on GPU
    depth_tensor = torch.from_numpy(depth_metric).to(device)

    # --- OPTIMIZATION: Perform all Disparity & Focus Math on the GPU ---
    
    # 1. Compute disparity (1/depth)
    safe_depth = torch.where(depth_tensor > 0.0, depth_tensor, torch.tensor(float('inf'), device=device))
    disp_tensor = 1.0 / safe_depth

    # 2. Depth map matches frame resolution
    disp_resized = disp_tensor

    # 3. Focus point logic
    tx = int(_click[0]) if _click[0] is not None else w // 2
    ty = int(_click[1]) if _click[1] is not None else h // 2
    tx = min(max(tx, 0), w - 1)
    ty = min(max(ty, 0), h - 1)

    disp_focus = disp_resized[ty, tx]

    # 4. Compute focus map natively on GPU
    defocus_abs = torch.abs(K_VALUE * (disp_resized - disp_focus))
    focus_map = 1.0 - torch.clamp(defocus_abs / MAX_COC, 0.0, 1.0)

    # --- Transfer minimal data to CPU for Visualization ---
    depth_np_viz = depth_metric
    focus_map_np = focus_map.detach().cpu().numpy()
    focus_uint8 = (focus_map_np * 255).astype(np.uint8)

    depth_at_click = float(depth_np_viz[ty, tx])

    if _log_click_values:
        print(
            f"Click ({tx}, {ty}) | Depth: {depth_at_click:.3f} m "
        )
        _log_click_values = False

    # --- Visualise ---
    depth_norm = cv2.normalize(depth_np_viz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    focus_colored = cv2.applyColorMap(focus_uint8, cv2.COLORMAP_INFERNO)

    # Draw UI elements
    dot_color = (0, 255, 0)
    for panel in (frame, depth_colored, focus_colored):
        cv2.circle(panel, (tx, ty), 8, dot_color, -1)
        cv2.circle(panel, (tx, ty), 8, (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,         "RGB",       (8, 24), font, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_colored, "Depth",     (8, 24), font, 0.7, (255, 255, 255), 2)
    cv2.putText(focus_colored, "Focus Map", (8, 24), font, 0.7, (255, 255, 255), 2)

    text_x = min(tx + 12, w - 220)
    text_y = max(ty - 12, 24)
    cv2.putText(
        depth_colored,
        f"d={depth_at_click:.2f}m",
        (text_x, text_y),
        font,
        0.55,
        (255, 255, 255),
        2,
    )

    # Tile and display
    combined = np.hstack([frame, depth_colored, focus_colored])
    cv2.imshow("Focus Map", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()