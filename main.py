import os
import certifi
import ssl
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms as T
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

# Load models
yolo = YOLO("yolov8n.pt")
yolo_classes = {
    0: "person", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 9: "traffic light"
}

deeplab = torch.hub.load('pytorch/vision:v0.13.1', 'deeplabv3_resnet101', pretrained=True).eval()
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lane mask function
def get_lane_mask(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    # Class 7 = road
    return (output_predictions == 7).astype(np.uint8) * 255

# Connect to BeamNG.tech
beamng = BeamNGpy('localhost', 64256)
scenario = Scenario('west_coast_usa', 'perception_combined')

vehicle = Vehicle('ego', model='etk800', licence='AUTO')
camera = Camera((0, 1.5, 1.5), (0, 0, 0), 70, (640, 480))
vehicle.attach_sensor('front_cam', camera)
scenario.add_vehicle(vehicle, pos=(394, 255, 118), rot=(0, 0, 45))
scenario.make(beamng)

beamng.open()
beamng.load_scenario(scenario)
beamng.start_scenario()

# Run perception loop
frame_id = 0
detections_log = []

try:
    while True:
        sensors = beamng.poll_sensors(vehicle)
        frame = sensors['front_cam']['colour'].copy()

        # Object Detection
        yolo_results = yolo(frame, verbose=False)[0]
        objects = []
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in yolo_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            objects.append({
                "class": yolo_classes[cls_id],
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
            # Draw (optional)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

        # Lane Detection
        lane_mask = get_lane_mask(frame)
        overlay = cv2.addWeighted(frame, 0.8, cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.2, 0)

        # Store frame data
        detections_log.append({
            "frame_id": frame_id,
            "objects": objects
        })

        # Show (optional)
        cv2.imshow("Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

finally:
    with open("combined_detections.json", "w") as f:
        json.dump(detections_log, f, indent=2)
    beamng.close()
    cv2.destroyAllWindows()