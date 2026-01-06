# Smart Face Attendance System

This is a Python-based application for managing attendance using Facial Recognition. It allows you to enroll users, train a recognition model, and mark attendance in real-time using a camera.

## Installation

1.  **Install Dependencies**:
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need `cmake` installed on your system for `dlib`.*

## How to Use

The system works in 4 sequential steps:

### 1. Enroll a New Person
Capture face images for a new user.
```bash
python enroll.py
```
-   Enter a unique **ID** (e.g., 101) and **Name**.
-   The system will capture 30 images.

### 2. Encode Faces
Process the captured images to extract face features.
```bash
python encode_faces.py
```

### 3. Train Model
Train the AI model to recognize the enrolled faces.
```bash
python train_model.py
```
*Note: Enroll at least 2 people for better training results.*

### 4. Start Recognition (Attendance)
Start the camera to detect faces and mark attendance.
```bash
python recognition.py
```
-   Press **'q'** or the **Exit** button to close the application.
-   Attendance is saved in `attendance.json`.

## Project Structure

-   **`enroll.py`**: User registration interface.
-   **`encode_faces.py`**: Processing engine for face images.
-   **`train_model.py`**: Machine learning model trainer.
-   **`recognition.py`**: Main application for real-time attendance.
-   **`config/`**: Contains system settings.
-   **`dataset/`**: Stores user face images.
-   **`output/`**: Stores trained models (`encodings.pickle`, `recognizer.pickle`).
