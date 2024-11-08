import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
 
cap = cv2.VideoCapture('video.mp4')

model = YOLO('license-plate-final-best.pt')
tracker = DeepSort(max_age=50)
reader = easyocr.Reader(['th','en'])

fps = cap.get(cv2.CAP_PROP_FPS)

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


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

    ocr_results = {}

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
            # cv2.putText(frame, f"{conf:0.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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


    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)

     # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
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