import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
import signal
import sys

# Initialize YOLOv10 model
yolov10_model = YOLOv10("./fake-yolo.pt")

# Set speed detection line, a straight line (x, y)
line_pts = [(0, 615), (1920, 615)]
# Initialize speed estimator
from ultralytics.solutions import speed_estimation, object_counter
speed_obj = speed_estimation.SpeedEstimator()
counter_obj = object_counter.ObjectCounter()

# Formula parameters for width and height
params_width = [-30843.21, 300.83]
params_height = [-29908.30, 284.79]

# Function to calculate box size based on fitted parameters
def calculate_box_size(y):
    width = np.polyval(params_width, y)
    height = np.polyval(params_height, y)
    return int(width), int(height)

# Create VideoWriter object
frame_width = 1920
frame_height = 1080
output_filename = "output.mp4"
video_writer = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
    30,  # FPS
    (frame_width, frame_height)  # Frame size
)

class YOLOv10Tracker:
    def __init__(self, yolov10_model):
        if not yolov10_model:
            raise ValueError("YOLOv10 model cannot be None")
        self.model = yolov10_model
        self.tracked_data = []
        self.frame = []
        self.detection_interval = 1  # Interval for stats (seconds)
        self.last_stat_time = time.time()
        self.vehicle_data = pd.DataFrame(columns=[
            'id', 'Time', 'type', 'x', 'y', 'xv', 'yv', 
            'yaw', 'length', 'width', 'prev_x', 'prev_y'
        ])  # Added prev_x and prev_y to store previous positions
        self.traffic_jam_threshold = 10  # Vehicle count threshold
        self.speed_threshold = 5  # Speed threshold (km/h)
        self.is_traffic_jam = False

    def get_frame(self):
        return self.frame

    def generate_statistics(self, frame):
        """Generate and display statistics"""
        if not self.vehicle_data.empty:
            # Count vehicles based on the current frame
            vehicle_count = len(self.vehicle_data)

            # Calculate average speed
            avg_speed = self.vehicle_data['xv'].mean()

            # Check for traffic jam condition: multiple vehicles and low average speed
            if vehicle_count >= self.traffic_jam_threshold and avg_speed <= self.speed_threshold:
                self.is_traffic_jam = True
                cv2.putText(frame, "Traffic Jam", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                self.is_traffic_jam = False

            # Display statistics in the top-left corner
            text_y = 50
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y += 40
            cv2.putText(frame, f"Average Speed: {avg_speed:.1f} km/h", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return vehicle_count, avg_speed
        else:
            # Return default values when vehicle data is empty
            return 0, 0

    def run_tracking(self, video_path):
        """Run YOLOv10 model for object detection"""
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Cannot open video file"

        while cap.isOpened():
            for _ in range(2):  # Discard the most recent 2 frames
                ret, self.frame = cap.read()
            if not ret:
                break

            # Data collection logic
            current_time = time.time()
            if current_time - self.last_stat_time >= self.detection_interval:
                self.last_stat_time = current_time

                # Generate and display statistics based on current vehicle data (per frame)
                self.generate_statistics(self.frame)

            # Perform YOLOv10 detection
            tracks = self.model.track(self.frame, conf=0.1, iou=0.1, persist=True, show=False, verbose=False)

            # Ensure tracks is not None
            if tracks is None:
                print("Error: Tracks data is None.")
                continue  # Skip this frame if no tracks

            # Temporary list to store detected vehicles for the current frame
            current_frame_vehicle_data = []

            # Draw detection results
            for result in tracks:
                boxes = result.boxes  # Get bounding box data
                confidences = boxes.conf  # Get confidence scores
                class_ids = boxes.cls  # Get class IDs
                names = result.names if hasattr(result, 'names') else {}

                # Iterate over each detection box
                for i, box in enumerate(boxes.xyxy):
                    x_min, y_min, x_max, y_max = box  # Get coordinates
                    confidence = confidences[i]  # Get current box confidence
                    class_id = int(class_ids[i])  # Get class ID

                    if confidence < 0.1:
                        continue

                    # Draw bounding box
                    color = (0, 255, 0)  # Green box
                    cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                    # Label the box with class and confidence
                    class_name = names[class_id] if names and class_id in names else "unknown"
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(self.frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Get the current position of the vehicle
                    current_position = (x_min + x_max) / 2, (y_min + y_max) / 2

                    # Get the previous position of the vehicle
                    prev_position = self.vehicle_data[self.vehicle_data['id'] == result[-1]][['prev_x', 'prev_y']].values

                    # Calculate speed if previous position exists
                    if len(prev_position) > 0:
                        prev_x, prev_y = prev_position[0]
                        speed_x = (current_position[0] - prev_x)  # Assuming pixel per frame distance
                        speed_y = (current_position[1] - prev_y)  # Assuming pixel per frame distance
                        speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
                    else:
                        speed = 0  # No previous position, so speed is 0

                    # Update vehicle data with current position and speed
                    new_data = {
                        'id': result[-1] if len(result) > 0 else -1,  # Use track_id
                        'Time': current_time,
                        'type': class_id, 
                        'x': current_position[0],
                        'y': current_position[1],
                        'xv': speed,  # Set speed
                        'yv': 0,  # Set speed in y-axis (if needed)
                        'yaw': 0,
                        'length': 6,  # Example conversion ratio
                        'width': 3,
                        'prev_x': current_position[0],  # Save current position as previous position for next frame
                        'prev_y': current_position[1]
                    }
                    current_frame_vehicle_data.append(new_data)

            # Update the vehicle data with the current frame's vehicle data
            self.vehicle_data = pd.DataFrame(current_frame_vehicle_data)

            # Show the current frame
            cv2.imshow("Frame", self.frame)
            video_writer.write(self.frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

def signal_handler(sig, frame):
    print("\nRecording stopped! Video saved as", output_filename)
    sys.exit(0)

# Capture Ctrl+C signal
signal.signal(signal.SIGINT, signal_handler)

# Start YOLOv10Tracker
url = '/mnt/sda1/shangao/山高数据2.20/拥堵_cleaned_renamed/拥堵10.mp4'
yolo_tracker = YOLOv10Tracker(yolov10_model)
yolo_tracker.run_tracking(url)
