import tkinter as tk
from tkinter import messagebox
from project.utils import Conf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def execute_training():
    """
    Loads encodings and trains a Support Vector Machine (SVM) classifier.
    """
    try:
        # Load Application Config
        app_config = Conf("config/config.json")
        path_encodings = app_config["encodings_path"]
        path_recognizer = app_config["recognizer_path"]
        path_label_encoder = app_config["le_path"]

        # 1. Load Face Encodings
        print("[STATUS] Loading face data from disk...")
        with open(path_encodings, "rb") as file_in:
            dataset = pickle.load(file_in)

        # 2. Encode Labels (Names -> Integers)
        print("[STATUS] Encoding labels...")
        label_enc = LabelEncoder()
        labels = label_enc.fit_transform(dataset["names"])

        # 3. Train the SVM Model
        print("[STATUS] Training SVM model...")
        recognizer_model = SVC(C=1.0, kernel="linear", probability=True)
        # Fit model on embeddings and numeric labels
        recognizer_model.fit(dataset["encodings"], labels)

        # 4. Save Trained Model
        print("[STATUS] Saving model to disk...")
        with open(path_recognizer, "wb") as file_out:
            pickle.dump(recognizer_model, file_out)

        # 5. Save Label Encoder
        with open(path_label_encoder, "wb") as file_out:
            pickle.dump(label_enc, file_out)

        # Success Feeback
        messagebox.showinfo(
            "Training Complete", "Machine Learning model trained successfully!"
        )
        close_app()

    except Exception as error:
        print(f"[ERROR] {error}")
        messagebox.showerror("Training Failed", f"An error occurred: {str(error)}")


def close_app():
    main_window.quit()


# --- GUI Setup ---
main_window = tk.Tk()
main_window.title("Model Trainer")
main_window.geometry("500x300")

# Center Window
w, h = 500, 300
src_w = main_window.winfo_screenwidth()
src_h = main_window.winfo_screenheight()
x = int((src_w - w) / 2)
y = int((src_h - h) / 2)
main_window.geometry(f"{w}x{h}+{x}+{y}")

main_window.config(bg="#f4f4f9")

# Header
lbl_header = tk.Label(
    main_window,
    text="Train Recognition Model",
    font=("Helvetica", 16, "bold"),
    bg="#f4f4f9",
)
lbl_header.pack(pady=10)

# Train Button
btn_train = tk.Button(
    main_window,
    text="Begin Training",
    command=execute_training,
    font=("Helvetica", 14),
    bg="#007BFF",
    fg="white",
)
btn_train.pack(pady=20)

main_window.mainloop()
