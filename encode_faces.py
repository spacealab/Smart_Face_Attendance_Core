import tkinter as tk
from tkinter import ttk, messagebox
from project.utils import Conf
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np


def run_face_encoding():
    """
    Main logic to read dataset images and generate 128-d face encodings.
    """
    try:
        # Load system configuration
        app_config = Conf("config/config.json")
        dataset_root = os.path.join(app_config["dataset_path"], app_config["class"])
        output_pickle_path = app_config["encodings_path"]

        # Retrieve all image paths
        all_image_paths = list(paths.list_images(dataset_root))
        total_count = len(all_image_paths)

        if total_count == 0:
            messagebox.showwarning(
                "No Data", "No images found in the dataset directory."
            )
            return

        # Storage for encodings and names
        known_encodings_list = []
        known_names_list = []

        # UI Progress Setup
        progress_bar["maximum"] = total_count

        for idx, img_path in enumerate(all_image_paths):
            # Update UI
            progress_bar["value"] = idx + 1
            lbl_status.config(text=f"Processing: {idx + 1}/{total_count}")
            main_window.update_idletasks()

            # Extract User ID/Name from directory structure
            person_name = img_path.split(os.path.sep)[-2]
            print(f"[LOG] Processing: {img_path} -> {person_name}")

            # Read and process image
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for specific model requirements (triplicated channels)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            img_prepared = np.expand_dims(img_gray, axis=2).repeat(3, axis=2)

            # Generate encodings
            found_encodings = face_recognition.face_encodings(img_prepared)

            for enc in found_encodings:
                known_encodings_list.append(enc)
                known_names_list.append(person_name)

        # Save results to pickle file
        data_dump = {"encodings": known_encodings_list, "names": known_names_list}
        with open(output_pickle_path, "wb") as file_handle:
            pickle.dump(data_dump, file_handle)

        messagebox.showinfo("Completed", f"Successfully encoded {total_count} images.")
        close_application()

    except Exception as error_msg:
        messagebox.showerror("System Error", f"An error occurred: {str(error_msg)}")


def close_application():
    main_window.quit()


# --- GUI Initialization ---
main_window = tk.Tk()
main_window.title("Face Encoder Utility")
main_window.geometry("500x300")

# Center window on screen
win_w, win_h = 500, 300
screen_w = main_window.winfo_screenwidth()
screen_h = main_window.winfo_screenheight()
x_pos = int((screen_w - win_w) / 2)
y_pos = int((screen_h - win_h) / 2)
main_window.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")

main_window.config(bg="#f4f4f9")

# Header
lbl_title = tk.Label(
    main_window,
    text="Face Encoding System",
    font=("Helvetica", 16, "bold"),
    bg="#f4f4f9",
)
lbl_title.pack(pady=10)

# Progress Components
progress_bar = ttk.Progressbar(main_window, length=400, mode="determinate")
progress_bar.pack(pady=20)

lbl_status = tk.Label(
    main_window, text="Ready to start...", font=("Helvetica", 12), bg="#f4f4f9"
)
lbl_status.pack()

# Buttons
btn_start = tk.Button(
    main_window,
    text="Start Encoding",
    command=run_face_encoding,
    font=("Helvetica", 14),
    bg="#007BFF",
    fg="white",
)
btn_start.pack(pady=20)

btn_exit = tk.Button(
    main_window,
    text="Close",
    command=close_application,
    font=("Helvetica", 14),
    bg="#FF4C4C",
    fg="white",
)
btn_exit.pack(pady=10)

main_window.mainloop()
