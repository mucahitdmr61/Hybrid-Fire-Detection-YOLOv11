from hybrid_inference import run_hybrid_inference
from ultralytics import YOLO
import joblib

yolo_model = YOLO("best.pt") 
xgb_model = joblib.load("XGBoost_Hybrid_Model_Final.pkl")

test_image = "WEB11314.jpg" 

if __name__ == "__main__":
    print("Hybrid Fire & Smoke Detection System is starting...")
    run_hybrid_inference(test_image, yolo_model, xgb_model)