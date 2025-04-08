# Wrong-lane-detection
Here’s a `README.md` file for the code you shared (`wrong_lane_only_for_one_side_road.py`). This documentation assumes the main goal is detecting wrong-lane vehicles using a YOLOv8 model on an RTSP stream:

---

```markdown
# 🚦 Wrong Lane Detection using YOLOv8

This project uses the YOLOv8 object detection model to identify vehicles moving in the **wrong direction** on a **one-sided road** by analyzing video from an RTSP stream.

## 📌 Features

- Detects vehicles (cars, trucks, buses, motorbikes) using YOLOv8.
- Tracks each vehicle using unique IDs.
- Determines whether a vehicle is moving in the correct or wrong direction based on its trajectory.
- Displays a bounding box (green for correct, red for wrong) with ID and timestamp.
- Optionally saves the processed video to disk.

## 🛠️ Requirements

Install dependencies via `pip`:

```bash
pip install opencv-python ultralytics numpy
```

You also need:
- Python 3.7+
- A trained YOLOv8 model (`.pt` file)
- An RTSP camera or stream URL

## 📂 File Structure

```
.
├── wrong_lane_only_for_one_side_road.py  # Main script for wrong lane detection
└── README.md                             # You're reading it!
```

## 🚀 Usage

Update the `main()` function inside `wrong_lane_only_for_one_side_road.py`:

```python
detector = WrongLaneDetector(model_path="path/to/your/yolov8n.pt")
rtsp_url = "rtsp://username:password@camera_ip:port/stream"
output_path = "optional_output_video.mp4"  # Or True to auto-generate filename
detector.process_rtsp_stream(rtsp_url, output_path)
```

Then run:

```bash
python wrong_lane_only_for_one_side_road.py
```

Press `q` to exit the video window.

## 🧠 How It Works

- The script tracks vehicle centroids over time.
- If a vehicle moves **top to bottom**, it's flagged as **wrong direction** (this can be adjusted).
- YOLOv8 class IDs used: 
  - 2: Car
  - 3: Motorcycle
  - 5: Bus
  - 7: Truck

## 📝 Notes

- `frame_skip` is set to 2 to process every other frame for performance.
- You can adjust `direction_points` to control how many recent points are analyzed to decide direction.
- Make sure the camera is mounted consistently for direction inference (e.g., always top-down for this logic to work).

## 📷 Output

Each detected vehicle will have:
- Bounding box
- Color indicating lane direction
- Unique ID
- Timestamp overlay
- Optional: output video saved to disk

## 📧 Contact

For questions or support, please reach out to the project contributor.

---

```

Let me know if you want a version tailored for `Jupyter Notebook` usage or Docker setup.
