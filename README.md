# Vehicle Detection and Counting using YOLO11

**YOLOv11 (You Only Look Once)** is a state-of-the-art object detection model known for its speed and accuracy. It uses deep learning techniques to efficiently detect and track objects in images and videos, making it ideal for real-time applications like vehicle counting and traffic monitoring.

This project implements **vehicle detection and counting** using **YOLOv11** and OpenCV. It processes a video file to track and count vehicles that cross a predefined red line, providing real-time visualizations of the detections and counts.

## 🚀 Features

- **Real-time vehicle detection and tracking** using YOLOv11.
- **Counts vehicles** that cross a red line in the video.
- **Bounding boxes and track IDs** displayed for each detected vehicle.
- **Video output with overlayed tracking results** is saved.

## 🛠️ Tech Stack

- **Python**
- **YOLOv11** (Ultralytics)
- **OpenCV** (Computer Vision Library)
- **PyTorch** (for YOLO model)
- **Numpy** (Array manipulations)

---

## 📂 Project Setup

### 1️⃣ Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required libraries:

```bash
pip install ultralytics opencv-python numpy torch torchvision torchaudio
```

### 2️⃣ Download YOLO11 Model

Download the **YOLO11 weights** file (`yolo11l.pt`) from [this link](https://docs.ultralytics.com/models/yolo11/#performance-metrics) , here in this link scroll down till you reach the  "🔥Performance" section and click on YOLO11l model to download the weights. Once downloaded, place it in the project directory.



### 3️⃣ Run the Project

```bash
python main.py
```

## 🎥 Input and Output

- **Input:** Video file (`./test videos/test video_1.mp4`)
- **Output:** Processed video saved as `output_video.mp4`
- **Visualization:** Displays the tracking results with bounding boxes and counts

## 📜 Code Explanation

1. **Loads YOLO model** using Ultralytics.
2. **Reads input video** and extracts properties like width, height, and FPS.
3. **Processes each frame** to detect and track vehicles (cars, bikes, etc.).
4. **Draws a red line** and counts vehicles crossing it.
5. **Saves processed video** with detected objects and counts.
6. **Displays real-time output** while processing.

## 🎯 Customization

- Change the input video path in `cap = cv2.VideoCapture('./test videos/test video_1.mp4')`.
- Modify `line_y_red = 430` to change the red line position.
- Adjust `classes=[1,2,3,5,6,7]` to track specific object categories:
  - `1` - Bicycle 🚲
  - `2` - Car 🚗
  - `3` - Motorcycle 🏍️
  - `5` - Bus 🚌
  - `6` - Train 🚆
  - `7` - Truck 🚛

## 📝 Future Improvements

- Add support for real-time webcam input.
- Implement speed estimation of detected vehicles.
- Export vehicle count data to a CSV file.

## 🤝 Contributing

Feel free to fork the repository, improve the project, and create a pull request!

## 📧 Contact

For any queries, reach out to me at **sruja2401@gmail.com**.

---

⚡ **Happy Coding!** 🚗🚦

