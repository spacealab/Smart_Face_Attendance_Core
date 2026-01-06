import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import pickle
import json
from tinydb import TinyDB, where
import face_recognition
from project.utils import Conf

# --- Initialization ---
app_config = Conf("config/config.json")

# Load trained models
recognizer_model = pickle.loads(open(app_config["recognizer_path"], "rb").read())
label_encoder = pickle.loads(open(app_config["le_path"], "rb").read())

# Database connections
database = TinyDB(app_config["db_path"])
users_table = database.table("student")

FILES_PATH = {"enroll": "database/enroll.json", "attendance": "attendance.json"}

# Camera Setup
video_stream = cv2.VideoCapture(0)


def mark_attendance_log(user_name, user_id):
    """
    Logs attendance if not already present for the current day.
    """
    if not user_name or str(user_name).lower() == "unknown":
        print("[LOG] Unknown user logged.")
        return

    # Load Database Files
    try:
        with open(FILES_PATH["enroll"], "r") as f:
            enroll_data = json.load(f)
    except FileNotFoundError:
        enroll_data = {"_default": {}, "student": {}}

    try:
        with open(FILES_PATH["attendance"], "r") as f:
            attendance_data = json.load(f)
    except FileNotFoundError:
        attendance_data = {"attendance": {}}

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check for duplicate entry today
    if user_id in attendance_data["attendance"]:
        last_record_date = (
            attendance_data["attendance"][user_id].get("date_time", "").split(" ")[0]
        )
        if last_record_date == today_str:
            return f"Already marked present today: {user_name} ({user_id})"

    # Record Attendance
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_data["attendance"][user_id] = {
        "name": user_name,
        "date_time": timestamp_str,
    }

    print(f"[SUCCESS] Attendance marked: {user_name} at {timestamp_str}")

    with open(FILES_PATH["attendance"], "w") as f:
        json.dump(attendance_data, f, indent=4)

    return None


# --- UI Application Class (Functional implementation) ---

root_window = tk.Tk()
root_window.title("Attendance System")
root_window.geometry("800x600")

lbl_status = tk.Label(root_window, text="Status: Waiting to Start", font=("Arial", 16))
lbl_status.pack(pady=20)

video_canvas = tk.Canvas(root_window, width=640, height=480)
video_canvas.pack()

# Global State Variables
g_prev_person = None
g_curr_person = None
g_consec_frames = 0
g_is_running = False


def process_video_frame():
    global g_prev_person, g_curr_person, g_consec_frames, g_is_running

    if not g_is_running:
        return

    success, frame = video_stream.read()
    if not success:
        print("[ERROR] Failed to read from camera.")
        return

    # Image Preprocessing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    # Model expects 3-channel input, so duplicate grayscale channels
    formatted_img = np.expand_dims(gray_frame, axis=2).repeat(3, axis=2)

    # Detect Faces
    detected_boxes = face_recognition.face_locations(
        formatted_img, model=app_config["detection_method"]
    )

    # Draw Boxes
    for top, right, bottom, left in detected_boxes:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    if detected_boxes:
        # Generate Embeddings
        face_encodings = face_recognition.face_encodings(rgb_frame, detected_boxes)

        # Predict Identity
        predictions = recognizer_model.predict_proba(face_encodings)[0]
        best_match_idx = np.argmax(predictions)
        g_curr_person = label_encoder.classes_[best_match_idx]

        # Stability Check (Debouncing)
        if g_prev_person == g_curr_person:
            g_consec_frames += 1
        else:
            g_consec_frames = 0
        g_prev_person = g_curr_person

        # Fetch Name from DB
        search_result = users_table.search(where(g_curr_person))
        if search_result:
            display_name = search_result[0][g_curr_person][0]
        else:
            display_name = f"Unknown ID: {g_curr_person}"

        # Overlay Text
        cv2.putText(
            frame,
            f"Identity: {display_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Log Attendance
        log_msg = mark_attendance_log(display_name, g_curr_person)
        if log_msg:
            lbl_status.config(text=log_msg)
        else:
            lbl_status.config(text=f"Detected: {display_name}")

    # Update UI Canvas
    final_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(final_img_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_image)

    video_canvas.create_image(0, 0, anchor="nw", image=tk_image)
    video_canvas.image = tk_image

    root_window.after(10, process_video_frame)


def on_start_click():
    global g_is_running
    g_is_running = True
    process_video_frame()


def on_exit_click():
    global g_is_running
    g_is_running = False
    video_stream.release()
    cv2.destroyAllWindows()
    root_window.quit()


# Control Buttons
btn_start = tk.Button(
    root_window,
    text="Start Camera",
    font=("Arial", 16),
    bg="#28a745",
    fg="white",
    command=on_start_click,
)
btn_start.pack(pady=10)

btn_exit = tk.Button(
    root_window,
    text="Exit Application",
    font=("Arial", 16),
    bg="#dc3545",
    fg="white",
    command=on_exit_click,
)
btn_exit.pack(pady=10)

root_window.mainloop()

# Cleanup on forced close
if video_stream.isOpened():
    video_stream.release()
cv2.destroyAllWindows()
