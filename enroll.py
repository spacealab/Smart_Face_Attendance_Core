import tkinter as tk
from tkinter import ttk, messagebox
from project.utils import Conf
from tinydb import TinyDB, where
import face_recognition
import cv2
import os
import time
import threading

# Ensure dataset directories exist
if not os.path.exists("dataset"):
    os.mkdir("dataset")
if not os.path.exists("dataset/PROJECT"):
    os.mkdir("dataset/PROJECT")

# Event flag to control the enrollment process
stop_signal = threading.Event()


def register_student():
    """
    Main function to handle user registration/enrollment.
    Captures images from the camera and correctly saves them.
    """
    stop_signal.clear()  # Reset signal

    # Disable button to prevent double-clicking
    btn_enroll.config(state=tk.DISABLED)

    # Get user inputs
    user_id = input_id.get().strip()
    user_name = input_name.get().strip()
    config_file_path = input_config_path.get().strip()

    # Input validation
    if not user_id or not user_name:
        messagebox.showerror("Validation Error", "Please provide both ID and Name.")
        btn_enroll.config(state=tk.NORMAL)
        return

    if not user_id.isdigit():
        messagebox.showerror("Validation Error", "ID must be numeric.")
        btn_enroll.config(state=tk.NORMAL)
        return

    # Load system configuration
    if not os.path.exists(config_file_path):
        messagebox.showerror(
            "File Error", f"Configuration file not found: '{config_file_path}'"
        )
        btn_enroll.config(state=tk.NORMAL)
        return

    app_config = Conf(config_file_path)

    # Connect to database
    database = TinyDB(app_config["db_path"])
    users_table = database.table("student")

    # Check if user already exists
    existing_users = []
    for record in users_table.all():
        for key, details in record.items():
            if user_id == key:
                existing_users.append(user_id)

    if existing_users:
        messagebox.showinfo(
            "Duplicate Entry", f"User ID '{existing_users[0]}' is already registered."
        )
        database.close()
        btn_enroll.config(state=tk.NORMAL)
        return

    # Inner function to run image capture in a separate thread
    def capture_faces():
        try:
            # Initialize camera
            video_stream = cv2.VideoCapture(0)

            # Define user folder path
            user_folder = os.path.join(
                app_config["dataset_path"], app_config["class"], user_id
            )
            os.makedirs(user_folder, exist_ok=True)

            img_count = 0
            required_count = app_config["face_count"]

            while img_count < required_count:
                if stop_signal.is_set():
                    messagebox.showinfo(
                        "Stopped", "Enrollment process interrupted by user."
                    )
                    break

                success, frame = video_stream.read()
                if not success:
                    break

                # Mirror frame and convert color space
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                detected_faces = face_recognition.face_locations(
                    rgb_frame, model=app_config["detection_method"]
                )
                display_frame = frame.copy()

                for top, right, bottom, left in detected_faces:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Add padding to face crop
                    pad = 70
                    y1 = max(0, top - pad)
                    y2 = min(frame.shape[0], bottom + pad)
                    x1 = max(0, left - pad)
                    x2 = min(frame.shape[1], right + pad)

                    face_crop = display_frame[y1:y2, x1:x2]

                    if img_count < required_count:
                        # Generate filename and save
                        filename = f"{str(img_count).zfill(5)}.png"
                        save_path = os.path.join(user_folder, filename)
                        cv2.imwrite(save_path, face_crop)
                        img_count += 1

                        # Update UI progress bar
                        main_window.after(
                            0, update_ui_progress, img_count, required_count
                        )

                # Show status on video
                cv2.putText(
                    frame,
                    "Status: Capturing...",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Enrollment Feed", frame)
                cv2.waitKey(1)

            # Cleanup resources
            video_stream.release()
            cv2.destroyAllWindows()

            if not stop_signal.is_set():
                # Save to DB if completed successfully
                users_table.insert({user_id: [user_name, "enrolled"]})
                messagebox.showinfo("Done", f"Successfully registered {user_name}.")
                clear_inputs()

        except Exception as err:
            messagebox.showerror("Runtime Error", f"An error occurred: {err}")
        finally:
            database.close()
            btn_enroll.config(state=tk.NORMAL)

    # Start capture thread
    threading.Thread(target=capture_faces, daemon=True).start()


def close_application():
    main_window.quit()


def update_ui_progress(current, total):
    progress_val = (current / total) * 100
    progress_bar["value"] = progress_val
    lbl_percentage.config(text=f"{int(progress_val)}%")


def interrupt_enrollment():
    stop_signal.set()
    messagebox.showinfo("Interrupted", "Stopping capture process...")


def clear_inputs():
    input_id.delete(0, tk.END)
    input_name.delete(0, tk.END)
    # Reset progress
    progress_bar["value"] = 0
    lbl_percentage.config(text="0%")
    messagebox.showinfo("Cleared", "Form reset.")


def render_background(canvas, w, h):
    canvas.delete("gradient_bg")
    for step in range(256):
        # Generate gradient color
        hex_color = f"#{int(0.6 * step):02x}{int(0.8 * step):02x}{step:02x}"
        y_start = int(step * h / 256)
        y_end = int((step + 1) * h / 256)
        canvas.create_rectangle(
            0, y_start, w, y_end, fill=hex_color, outline="", tags="gradient_bg"
        )


# --- GUI Setup ---
main_window = tk.Tk()
main_window.title("User Enrollment System")
main_window.geometry("800x600")
main_window.configure(bg="#eef2f3")

# Background Canvas
bg_canvas = tk.Canvas(main_window, highlightthickness=0)
bg_canvas.pack(fill="both", expand=True)


def handle_resize(event):
    render_background(bg_canvas, event.width, event.height)


bg_canvas.bind("<Configure>", handle_resize)

# Header
lbl_header = tk.Label(
    main_window,
    text="Face Enrollment",
    font=("Helvetica", 22, "bold"),
    bg="#ff0000",
    fg="white",
)
lbl_header.place(relx=0.5, rely=0.05, anchor="n", width=400)

# Input Container
frame_inputs = tk.Frame(
    main_window, bg="#ffffff", padx=20, pady=20, relief="solid", bd=2
)
frame_inputs.place(relx=0.5, rely=0.4, anchor="center", relwidth=0.6, relheight=0.5)


def make_input_field(parent, label_txt, default_val=""):
    lbl = tk.Label(parent, text=label_txt, font=("Helvetica", 12), bg="#ffffff")
    lbl.pack(anchor="w", pady=5)
    field = tk.Entry(parent, font=("Helvetica", 14))
    field.insert(0, default_val)
    field.pack(fill="x", pady=5)
    return field


input_id = make_input_field(frame_inputs, "User ID:")
input_name = make_input_field(frame_inputs, "Full Name:")

# Config path (readonly)
input_config_path = make_input_field(
    frame_inputs, "Config File:", default_val="config/config.json"
)
input_config_path.config(state=tk.DISABLED)

# Progress Bar
progress_bar = ttk.Progressbar(frame_inputs, length=300, mode="determinate")
progress_bar.pack(pady=20)
lbl_percentage = tk.Label(frame_inputs, text="0%", font=("Helvetica", 14), bg="#ffffff")
lbl_percentage.pack(pady=10)

# Buttons
frame_btns = tk.Frame(main_window, bg="#eef2f3")
frame_btns.place(relx=0.5, rely=0.8, anchor="center")

btn_enroll = tk.Button(
    frame_btns,
    text="Enroll",
    font=("Helvetica", 14, "bold"),
    bg="#ff0000",
    fg="white",
    command=register_student,
)
btn_enroll.pack(side=tk.LEFT, padx=10)

btn_stop = tk.Button(
    frame_btns,
    text="Stop",
    font=("Helvetica", 14, "bold"),
    bg="#ff0000",
    fg="white",
    command=interrupt_enrollment,
)
btn_stop.pack(side=tk.LEFT, padx=10)

btn_reset = tk.Button(
    frame_btns,
    text="Reset",
    font=("Helvetica", 14, "bold"),
    bg="#ff0000",
    fg="white",
    command=clear_inputs,
)
btn_reset.pack(side=tk.LEFT, padx=10)

btn_exit = tk.Button(
    main_window,
    text="Exit",
    command=close_application,
    font=("Helvetica", 14),
    bg="#ff0000",
    fg="white",
)
btn_exit.pack(pady=10)

# Styling
style = ttk.Style(main_window)
style.theme_use("default")
style.configure(
    "TProgressbar", troughcolor="#e0e0e0", background="#ff0000", thickness=20
)

main_window.mainloop()
