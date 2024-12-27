import cv2
import math
import time
import torch
import xlsxwriter
import pandas as pd
import numpy as np
from numpy.matlib import empty


from ultralytics import YOLOv10
from utils.object_tracking_class import ObjectTracking
objectTracking = ObjectTracking()
deepsort = objectTracking.initialize_deepsort()


#change variables as needed
videoName = "chicken2"
videoExtention = ".mkv"
numChickens = 7

id_remap = {}


#create a excel file for the data
workbook = xlsxwriter.Workbook(videoName)
worksheet = workbook.add_worksheet()


cap = cv2.VideoCapture("Resources/"+videoName+videoExtention)
model = YOLOv10("runs/detect/train6/weights/last.pt")
df = pd.DataFrame(columns=["time","id","center_x","center_y"])

arrayIds = np.arange(start=1, stop=numChickens+1, step=1)
startIds = set(arrayIds)

ctime = 0
ptime = 0
count = 0
oldId = list()
while True:
    xywh_bboxs = []
    confs = []
    oids = oldId.copy()
    outputs = []
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    CenterBox_height = int(0.2*frame_height)
    CenterBox_width = int(0.2*frame_width)
    if ret:

        ctime = time.time()
        count += 1
        print(f"Frame Count: {count}")
        results = model.predict(frame, conf = 0.8)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = math.ceil(box.conf[0] * 100) / 100
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #center coordinates of the bouding boxes
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                #height and width of the bounding boxes
                bbox_width = abs(x1 - x2)
                bbox_height = abs(y1 - y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                xywh_bboxs.append(xcycwh)
                confs.append(conf)
                classNameInt = int(box.cls[0])
                oids.append(classNameInt)

        xywhs = torch.tensor(xywh_bboxs)
        confidence = torch.tensor(confs)

        if xywhs.nelement() != 0:
            outputs = deepsort.update(xywhs, confidence, oids, frame)

            #pd.DataFrame(outputs).to_csv("outputs/"+videoName+".csv", index=False)


        current_ids = set()
        if len(outputs) > 0:

            bbox_xyxy = outputs[:,:4]
            identities = outputs[:,-2]
            classID = outputs[:,-1]
            centerX = (bbox_xyxy[:,0]+bbox_xyxy[:,2])/2
            centerY = (bbox_xyxy[:,1]+bbox_xyxy[:,3])/2

            current_ids = set(identities)
            invalid_ids = current_ids - startIds
            print(invalid_ids)
            missing_ids = list(startIds - current_ids)
            for invalid_id in invalid_ids:
                if missing_ids:
                    new_id = missing_ids.pop(0)
                    id_remap[invalid_id] = new_id

            identities = [id_remap.get(i, i) for i in outputs[:, -2]]

            for id, cx, cy in zip(identities, centerX, centerY):
                df = pd.concat([df, pd.DataFrame([[ctime, id, cx.item(), cy.item()]], columns=df.columns)],ignore_index=True)

            objectTracking.draw_boxes(frame, bbox_xyxy, identities, classID)







        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.putText(frame, f"Frame Count: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

df.to_excel(f"{videoName}.xlsx", index=False)
print(f"Data saved to {videoName}.xlsx")

