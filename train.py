from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
# model = YOLO("yolo11n.pt")
model =YOLO( "weights/fake-yolo.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="./data/shangao.yaml", epochs=600, imgsz=640,batch=4)

# Save the trained model
# model.export(format="pt")