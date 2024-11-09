import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import datetime
from collections import OrderedDict
 
cap = cv2.VideoCapture(r'C:\Users\calvi\OneDrive\Desktop\project\ml_project\tobeit\tobeit-68-alpr\2103099-hd_1920_1080_30fps.mp4')

# Initialize YOLO model
model = YOLO(r'C:\Users\calvi\OneDrive\Desktop\project\ml_project\tobeit\tobeit-68-alpr\license-plate-final-best.pt')
reader = easyocr.Reader(['en'])

fps = cap.get(cv2.CAP_PROP_FPS)

print("A")
CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

ocr_results = OrderedDict()

def run_ocr(plate, trackId):
    # if ocr_results.get(trackId):
    #     ocr_results.move_to_end(trackId, last=False)
    #     return ocr_results[trackId]
    try:
        # crop_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('plate', crop_gray)
        results = reader.readtext(plate)
        if len(results):
            print(results)
        final_text = []
        for result in results:
            _, text, score = result
            if float(score) > 0.5:
                final_text.append(text)
        
        final_text = " ".join(final_text)
        if len(final_text) > 1:
            # if len(ocr_results) > 20:
            #     ocr_results.popitem(last=False)
            # ocr_results[trackId] = final_text
            return final_text
        
        return None
    except Exception as e:
        return None

def process_frame():
    start = datetime.datetime.now()
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return False

    # Run YOLO with tracking enabled
    results = model.track(frame, conf=CONFIDENCE_THRESHOLD, persist=True, tracker="botsort.yaml")[0]
    
    # Process tracked detections
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        
        # Draw boxes and process each detection
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get the license plate region
            plate = frame[y1:y2, x1:x2, :]
            if plate.size < 1:
                continue

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), GREEN, -1)
            cv2.putText(frame, str(track_id), (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            
            # Run OCR on the plate
            text = run_ocr(plate, track_id)
            if text:
                cv2.putText(frame, text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # Calculate and display FPS
    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        return False

    return True
    
while cap.isOpened():
    if not process_frame():
        break
print("B")
cap.release()
cv2.destroyAllWindows()

print("C")