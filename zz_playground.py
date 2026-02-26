import cv2
import torch
import numpy as np
from datetime import datetime
from depth_anything_3.api import DepthAnything3

# 1. Load Depth Anything 3 Metric Large model
model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
model.eval()

# --- Focus settings ---
# Physical camera defaults (full-frame style)
FOCAL_MM_DEFAULT = 35.0
FSTOP_DEFAULT = 2.8
SENSOR_WIDTH_MM_DEFAULT = 36.0
MAX_COC_UM_DEFAULT = 30.0
PANEL_WIDTH = 540
PANEL_HEIGHT = 400

# --- Inference settings ---
PROCESS_RES = 518
WINDOW_NAME = "DA3 Playground"
CAMERA_INDEX = -1

_click = [None, None]
_panel_w_for_click = PANEL_WIDTH
_panel_h_for_click = PANEL_HEIGHT

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click[0] = x % _panel_w_for_click
        _click[1] = y % _panel_h_for_click


def _noop(_):
    return


def _add_gaussian_noise(image_bgr, sigma):
    if sigma <= 0:
        return image_bgr.copy()
    noise = np.random.normal(0.0, sigma, image_bgr.shape).astype(np.float32)
    noisy = image_bgr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_blur(image_bgr, blur_level):
    if blur_level <= 0:
        return image_bgr.copy()
    kernel_size = int(2 * blur_level + 1)
    return cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)


def _infer_depth_metric(input_bgr, focal_px):
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    prediction = model.inference(
        [input_rgb],
        process_res=PROCESS_RES,
        process_res_method="upper_bound_resize",
    )
    depth_canonical = prediction.depth[0]
    return (focal_px * depth_canonical) / 300.0


def _compute_focus_map_from_lens(depth_metric_m, focus_distance_m, focal_mm, fstop, max_coc_um):
    eps = 1e-6
    depth_mm = np.maximum(depth_metric_m * 1000.0, eps)
    focus_mm = max(focus_distance_m * 1000.0, focal_mm + eps)
    fstop = max(fstop, eps)

    coc_mm = (focal_mm * focal_mm / (fstop * max(focus_mm - focal_mm, eps))) * np.abs(
        (depth_mm - focus_mm) / depth_mm
    )

    max_coc_mm = max(max_coc_um / 1000.0, eps)
    focus_map = 1.0 - np.clip(coc_mm / max_coc_mm, 0.0, 1.0)
    return focus_map, coc_mm


def _capture_base_frame():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture a frame from camera")

    captured = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT))
    captured = cv2.rotate(captured, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return captured


base_frame = _capture_base_frame()
base_h, base_w = base_frame.shape[:2]

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

cv2.createTrackbar("Focal mm x10", WINDOW_NAME, int(FOCAL_MM_DEFAULT * 10), 2000, _noop)
cv2.createTrackbar("F-stop x10", WINDOW_NAME, int(FSTOP_DEFAULT * 10), 220, _noop)
cv2.createTrackbar("SensorW mm x10", WINDOW_NAME, int(SENSOR_WIDTH_MM_DEFAULT * 10), 600, _noop)
cv2.createTrackbar("Max CoC um", WINDOW_NAME, int(MAX_COC_UM_DEFAULT), 200, _noop)
cv2.createTrackbar("Blur", WINDOW_NAME, 0, 30, _noop)
cv2.createTrackbar("Noise", WINDOW_NAME, 0, 100, _noop)

cache = {
    "blur": None,
    "noise": None,
    "rows": None,
}

last_combined = None

while True:
    focal_mm = max(1.0, cv2.getTrackbarPos("Focal mm x10", WINDOW_NAME) / 10.0)
    fstop = max(0.7, cv2.getTrackbarPos("F-stop x10", WINDOW_NAME) / 10.0)
    sensor_width_mm = max(1.0, cv2.getTrackbarPos("SensorW mm x10", WINDOW_NAME) / 10.0)
    max_coc_um = max(1.0, float(cv2.getTrackbarPos("Max CoC um", WINDOW_NAME)))
    blur_level = cv2.getTrackbarPos("Blur", WINDOW_NAME)
    noise_sigma = cv2.getTrackbarPos("Noise", WINDOW_NAME)

    focal_px = (focal_mm / sensor_width_mm) * base_w

    if cache["rows"] is None or cache["blur"] != blur_level or cache["noise"] != noise_sigma:
        clean_bgr = base_frame.copy()
        noisy_bgr = _add_gaussian_noise(base_frame, noise_sigma)
        blurry_bgr = _apply_blur(base_frame, blur_level)
        noisy_blurry_bgr = _apply_blur(noisy_bgr, blur_level)

        variants = [
            ("Clean", clean_bgr),
            ("Noisy", noisy_bgr),
            ("Blurry", blurry_bgr),
            ("Noisy + Blurry", noisy_blurry_bgr),
        ]

        rows = []
        for row_name, variant_bgr in variants:
            depth_metric = _infer_depth_metric(variant_bgr, focal_px)

            depth_h, depth_w = depth_metric.shape
            variant_resized = cv2.resize(variant_bgr, (depth_w, depth_h))

            rows.append(
                {
                    "name": row_name,
                    "input": variant_resized,
                    "depth_metric": depth_metric,
                }
            )

        cache["rows"] = rows
        cache["blur"] = blur_level
        cache["noise"] = noise_sigma

    rendered_rows = []
    for row in cache["rows"]:
        input_panel = row["input"].copy()
        depth_metric = row["depth_metric"]

        h, w = depth_metric.shape
        _panel_w_for_click = w
        _panel_h_for_click = h

        tx = int(_click[0]) if _click[0] is not None else w // 2
        ty = int(_click[1]) if _click[1] is not None else h // 2
        tx = min(max(tx, 0), w - 1)
        ty = min(max(ty, 0), h - 1)

        safe_depth = np.where(depth_metric > 0.0, depth_metric, np.inf)
        focus_distance = float(safe_depth[ty, tx])
        focus_map, coc_mm = _compute_focus_map_from_lens(
            safe_depth,
            focus_distance,
            focal_mm,
            fstop,
            max_coc_um,
        )

        focus_value = float(focus_map[ty, tx])
        depth_value = float(depth_metric[ty, tx])
        coc_um = float(coc_mm[ty, tx] * 1000.0)

        depth_norm = cv2.normalize(depth_metric, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        focus_uint8 = (focus_map * 255.0).astype(np.uint8)
        focus_colored = cv2.applyColorMap(focus_uint8, cv2.COLORMAP_INFERNO)

        dot_color = (0, 255, 0)
        for panel in (input_panel, depth_colored, focus_colored):
            cv2.circle(panel, (tx, ty), 8, dot_color, -1)
            cv2.circle(panel, (tx, ty), 8, (0, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(input_panel, f"{row['name']} Input", (8, 24), font, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"Depth d={depth_value:.2f}m", (8, 24), font, 0.7, (255, 255, 255), 2)
        cv2.putText(
            focus_colored,
            f"Focus={focus_value:.2f} CoC={coc_um:.1f}um",
            (8, 24),
            font,
            0.7,
            (255, 255, 255),
            2,
        )

        rendered_rows.append(np.hstack([input_panel, depth_colored, focus_colored]))

    combined = np.vstack(rendered_rows)
    cv2.putText(
        combined,
        (
            "Keys: q=quit, r=recapture, c=capture | "
            f"f={focal_mm:.1f}mm N={fstop:.1f} sensorW={sensor_width_mm:.1f}mm"
        ),
        (8, combined.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    last_combined = combined.copy()
    cv2.imshow(WINDOW_NAME, combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        base_frame = _capture_base_frame()
        base_h, base_w = base_frame.shape[:2]

        cache["rows"] = None
        cache["blur"] = None
        cache["noise"] = None

    if key == ord('c') and last_combined is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"playground_capture_{timestamp}.png"
        cv2.imwrite(out_path, last_combined)
        print(f"Saved playground capture: {out_path}")

cv2.destroyAllWindows()