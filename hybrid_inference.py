import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_hybrid_inference(image_path, yolo_model, xgb_model):
    """
    Performs hybrid fire detection by combining YOLOv11 and XGBoost.
    
    Args:
        image_path (str): Path to the input image.
        yolo_model: Loaded YOLOv11 model instance.
        xgb_model: Loaded XGBoost classifier instance.
    """
    # 1. YOLO Prediction
    results = yolo_model.predict(image_path, conf=0.15, imgsz=768, verbose=False)[0]

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Prepare visualization views
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yolo_view = img_rgb.copy()
    hybrid_view = img_rgb.copy()

    thickness, f_scale, t_thick = 2, 0.6, 2
    class_names = ['Smoke', 'Fire']

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_cls = int(box.cls[0])
        yolo_conf = float(box.conf[0])
        yolo_name = class_names[yolo_cls]

        # Crop Region of Interest (ROI)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: continue

        # --- YOLO VISUALIZATION (LEFT PANEL) ---
        yolo_color = (255, 0, 0) if yolo_name == "Fire" else (0, 255, 255)
        cv2.rectangle(yolo_view, (x1, y1), (x2, y2), yolo_color, thickness)
        cv2.putText(yolo_view, f"{yolo_name} {yolo_conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, yolo_color, t_thick)

        # --- HYBRID ANALYSIS (XGBoost) ---
        # Feature Extraction: HSV Color Space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_roi)[:3]
        
        # Feature Extraction: Texture (Laplacian Variance)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

        # Build Feature Vector (Must match training order)
        # [Conf, H, S, V, Edge_Score, Aspect_Ratio, Area]
        feat = np.array([[
            yolo_conf, mean_hsv[0], mean_hsv[1], mean_hsv[2],
            edge_score, (x2-x1)/(y2-y1) if (y2-y1)>0 else 0, (x2-x1)*(y2-y1)
        ]])

        # XGBoost Probabilities
        probs = xgb_model.predict_proba(feat)[0]
        fire_prob = probs[1]
        smoke_prob = probs[0]

        # --- HYBRID DECISION LOGIC ---
        if fire_prob > 0.5:
            final_name = "Fire"
            final_prob = fire_prob
            final_color = (255, 69, 0) # Orange/Red
        else:
            final_name = "Smoke"
            final_prob = smoke_prob
            final_color = (0, 255, 0) # Green

        # --- HYBRID VISUALIZATION (RIGHT PANEL) ---
        label = f"{final_name} %{final_prob*100:.1f}"
        cv2.rectangle(hybrid_view, (x1, y1), (x2, y2), final_color, thickness)
        cv2.putText(hybrid_view, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale, final_color, t_thick)

    # --- PLOTTING OUTPUT ---
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(yolo_view)
    plt.title("Standard YOLOv11 Detection", fontsize=14)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hybrid_view)
    plt.title("Hybrid Decision (YOLO + XGBoost Layer)", fontsize=14, color="darkgreen")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Placeholder for model loading
    # yolo_model = YOLO("path/to/best.pt")
    # xgb_model = load_your_xgb_model()
    print("Inference script loaded. Use 'run_hybrid_inference()' to process images.")
