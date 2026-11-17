import io
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ultralytics import YOLO
import tensorflow as tf

# Initialize FastAPI Application
app = FastAPI(
    title="Multi-Modal Disaster Response API",
    description="Production-grade pipeline utilizing 2-Stage CV and Fine-Tuned Quantized NLP for incident analysis.",
    version="2.1.0"
)

# =====================================================================
# GLOBAL MODEL INITIALIZATION (Loaded once on server startup)
# =====================================================================

try:
    # 1. Computer Vision Stage 1: Bounding Box Detection
    print("Initializing Stage 1 CV: YOLOv8 Nano...")
    yolo_model = YOLO("yolov8n.pt") 

    # 2. Computer Vision Stage 2: Structural Damage Classification
    print("Initializing Stage 2 CV: EfficientNetV2-B0...")
    efficientnet_model = tf.keras.models.load_model("damage_model.h5")

    # 3. Fine-Tuned NLP: Disaster Type Classification
    print("Initializing Fine-Tuned NLP: Local DistilBERT...")
    nlp_path = "./nlp_model"
    nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
    base_nlp_model = AutoModelForSequenceClassification.from_pretrained(nlp_path)

    # Apply INT8 dynamic quantization to accelerate CPU execution and lower RAM usage
    print("Applying INT8 Dynamic Quantization to NLP model...")
    quantized_nlp_model = torch.quantization.quantize_dynamic(
        base_nlp_model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    nlp_classifier = pipeline(
        "text-classification", 
        model=quantized_nlp_model, 
        tokenizer=nlp_tokenizer
    )

    # 4. Named Entity Recognition (NER) for Geographic Extraction
    print("Initializing NER Pipeline...")
    ner_extractor = pipeline(
        "ner", 
        model="dbmdz/bert-large-cased-finetuned-conll03-english", 
        aggregation_strategy="simple"
    )
    print("All models successfully loaded into memory.")

except Exception as init_error:
    print(f"CRITICAL ERROR DURING MODEL INITIALIZATION: {init_error}")
    raise init_error


# =====================================================================
# UTILITY SCHEMAS
# =====================================================================

class TextPayload(BaseModel):
    text: str


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/")
async def root():
    return {"status": "online", "pipeline": "Multi-Modal Disaster Response Engine v2.1"}

# ---------------------------------------------------------------------
# UTILITY 1: Text-Only Analysis
# ---------------------------------------------------------------------
@app.post("/analyze-text")
async def analyze_text(payload: TextPayload):
    try:
        entities = ner_extractor(payload.text)
        locations = list(set([ent['word'] for ent in entities if 'LOC' in ent['entity_group']]))
        
        nlp_result = nlp_classifier(payload.text)[0]
        
        return {
            "disaster_type": nlp_result['label'],
            "confidence": round(float(nlp_result['score']), 4),
            "locations_found": locations
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------
# UTILITY 2: Image-Only Analysis (2-Stage Pipeline)
# ---------------------------------------------------------------------
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        yolo_results = yolo_model(image)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        
        crops = []
        is_damaged = False
        
        if len(boxes) > 0:
            valid_vehicle_classes = [2, 3, 5, 7]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(box[5])
                if class_id in valid_vehicle_classes:
                    cropped_img = image.crop((x1, y1, x2, y2))
                    crops.append(cropped_img)
        else:
            crops.append(image)
            
        for cropped_img in crops:
            resized_crop = cropped_img.resize((224, 224))
            img_array = np.array(resized_crop) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            cv_pred = efficientnet_model.predict(img_array, verbose=0)
            # THE FIX: Increased threshold from 0.5 to 0.85 to filter out clouds/forests
            if float(cv_pred[0][0]) > 0.95:
                is_damaged = True
                break 

        return {
            "structures_analyzed": len(crops),
            "structural_damage_detected": is_damaged
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------
# MASTER ORCHESTRATION: Multi-Modal Fusion Logging
# ---------------------------------------------------------------------
@app.post("/submit-incident")
async def submit_incident(text_report: str, file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        # =============================================================
        # STEP 1: TWO-STAGE COMPUTER VISION PROCESSING
        # =============================================================
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        yolo_results = yolo_model(image)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        
        crops = []
        is_damaged = False
        
        if len(boxes) > 0:
            valid_vehicle_classes = [2, 3, 5, 7]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(box[5])
                if class_id in valid_vehicle_classes:
                    cropped_img = image.crop((x1, y1, x2, y2))
                    crops.append(cropped_img)
        else:
            crops.append(image)
                
        for cropped_img in crops:
            resized_crop = cropped_img.resize((224, 224))
            img_array = np.array(resized_crop) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            cv_pred = efficientnet_model.predict(img_array, verbose=0)
            # THE FIX: Increased threshold from 0.5 to 0.85 to filter out clouds/forests
            if float(cv_pred[0][0]) > 0.95:
                is_damaged = True
                break 

        # =============================================================
        # STEP 2: FINE-TUNED NLP PROCESSING
        # =============================================================
        entities = ner_extractor(text_report)
        locations = list(set([ent['word'] for ent in entities if 'LOC' in ent['entity_group']]))
        
        nlp_result = nlp_classifier(text_report)[0]
        disaster_type = nlp_result['label']
        nlp_confidence = float(nlp_result['score'])
        
        if nlp_confidence < 0.50:
            disaster_type = "unverified"

        # =============================================================
        # STEP 3: MULTI-MODAL FUSION LOGIC
        # =============================================================
        is_disaster_confirmed = disaster_type not in ["normal", "unverified"]
        
        if is_damaged and is_disaster_confirmed:
            severity = "CRITICAL"
        elif is_damaged or is_disaster_confirmed:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        # =============================================================
        # STEP 4: RESPONSE GENERATION
        # =============================================================
        return {
            "incident_summary": {
                "inferred_disaster": disaster_type,
                "nlp_confidence_score": round(nlp_confidence, 4),
                "verified_structural_damage": is_damaged,
                "impacted_areas": locations
            },
            "system_action": {
                "calculated_severity": severity,
                "dispatch_required": severity in ["CRITICAL", "MEDIUM"]
            }
        }
        
    except Exception as e:
        return {"error": f"Internal server error processing payload: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)