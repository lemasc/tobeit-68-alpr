import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
from collections import OrderedDict
 
cap = cv2.VideoCapture(r'D:\Coding\train-poon-poon\alpr\2103099-hd_1920_1080_30fps.mp4')

model = YOLO(r'D:\Coding\train-poon-poon\alpr\license-plate-final-best.pt')
tracker = DeepSort(max_age=50)
reader = easyocr.Reader(['en'])

fps = cap.get(cv2.CAP_PROP_FPS)

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

ocr_results = OrderedDict()

def run_ocr(plate, trackId, plateConfidence = 0):
    existing = ocr_results.get(trackId)
    if existing and plateConfidence and existing["confidence"] > plateConfidence:
        ocr_results.move_to_end(trackId, last=False)
        return ocr_results[trackId]["text"]
    try:
        crop_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, crop_thresh = cv2.threshold(crop_gray, 70, 255, cv2.THRESH_BINARY)
        # cv2.imshow('plate', crop_thresh)
        results = reader.readtext(crop_thresh)
        final_text = []
        for result in results:
            _, text, score = result
            final_text.append(text)
        
        if len(final_text) >= 1:
            final_text = " ".join(final_text)
            if len(ocr_results) > 20:
                ocr_results.popitem(last=False)
            if plateConfidence is None:
                plateConfidence = 0
            ocr_results[trackId] = {
                "confidence": plateConfidence,
                "text": final_text
            }
            return final_text
        
        return None
    except Exception as e:
        return None


def process_frame():
    start = datetime.datetime.now()
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return False
    

    # run the YOLO model on the frame
    detections = model(frame)[0]

    
    # initialize the list of bounding boxes and confidences
    results = []

    for d in detections:
        # print("-----------------")
        for boxes, conf, classId in zip(d.boxes.xyxy.tolist(), d.boxes.conf.tolist(), d.boxes.cls.tolist()):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(x) for x in boxes]
            w = x2 - x1
            h = y2 - y1

            results.append([[x1, y1, w, h], conf, classId])

            # Print the confidence over the rectangle
            cv2.putText(frame, f"{conf:0.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the license plate
            # plate = frame[y1:y2, x1:x2]
            # cv2.imshow('frame',frame)
            # cv2.imshow('plate', plate)
            # Wait for window to close
            # cv2.waitKey(60 * 1000)
            # Run the OCR on the given plate
            

        # return False
        # print(list(result.boxes.xyxy))
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # # update the tracker with the new detections
    # tracks = tracker.update_tracks(results, frame=frame)

    #  # loop over the tracks
    # for track in tracks:
        # # if the track is not confirmed, ignore it
        # if not track.is_confirmed():
        #     continue

        # # get the track id and the bounding box
        # track_id = track.track_id
        # ltrb = track.to_ltrb()

        # xmin, ymin, xmax, ymax = [int(x) for x in ltrb]

        # plate = frame[ymin:ymax, xmin:xmax, :]
        # if plate.size < 1:
        #     continue

        
        # text = run_ocr(plate, track_id, track.det_conf)

        # if text:
        #     # print(f"Track ID: {track_id}, Text: {text}")

        #     # draw the bounding box and the track id
        #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        #     cv2.putText(frame, text, (xmin + 5, ymin - 8),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
        # print(text)
        # if text:
        #     cv2.putText(frame, text[0], (xmin, ymin - 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
        
    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    # print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        return False

    return True
    
while cap.isOpened():
    if not process_frame():
        break
    
 
cap.release()
cv2.destroyAllWindows()