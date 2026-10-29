"""Simple desktop YOLO object detector: webcam + image upload."""
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
PREVIEW_SIZE = (920, 570)


class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detector")
        self.root.geometry("1020x760")
        self.root.minsize(760, 630)
        self.root.configure(bg="#f1f5f9")
        self.model = None
        self.camera = None
        self.webcam_running = False
        self.last_result = None
        self.photo = None
        self.confidence = tk.DoubleVar(value=0.40)
        self.status = tk.StringVar(value="Ready. Choose an image or start webcam.")
        self.objects = tk.StringVar(value="Objects detected: —")

        header = tk.Frame(root, bg="#0f172a", padx=24, pady=17)
        header.pack(fill="x")
        tk.Label(header, text="YOLO Object Detector", bg="#0f172a", fg="white",
                 font=("Arial", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="A beginner-friendly object detection project using real YOLO inference",
                 bg="#0f172a", fg="#cbd5e1", font=("Arial", 11)).pack(anchor="w", pady=(5, 0))

        tools = tk.Frame(root, bg="#f1f5f9", padx=22, pady=14)
        tools.pack(fill="x")
        self.upload_button = ttk.Button(tools, text="Upload Image", command=self.open_image)
        self.upload_button.pack(side="left", padx=(0, 10))
        self.start_button = ttk.Button(tools, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(side="left", padx=(0, 10))
        self.stop_button = ttk.Button(tools, text="Stop Webcam", command=self.stop_webcam, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 10))
        self.save_button = ttk.Button(tools, text="Save Result", command=self.save_result, state="disabled")
        self.save_button.pack(side="left")

        settings = tk.Frame(root, bg="#f1f5f9", padx=22)
        settings.pack(fill="x")
        tk.Label(settings, text="Confidence threshold", bg="#f1f5f9").pack(side="left")
        tk.Scale(settings, from_=0.1, to=0.9, resolution=0.05, orient="horizontal",
                 variable=self.confidence, bg="#f1f5f9", highlightthickness=0,
                 length=170).pack(side="left", padx=(8, 10))
        tk.Label(settings, textvariable=self.objects, bg="#f1f5f9", fg="#334155",
                 font=("Arial", 10)).pack(side="left")

        display_frame = tk.Frame(root, bg="#e2e8f0", padx=8, pady=8)
        display_frame.pack(fill="both", expand=True, padx=22, pady=(10, 12))
        self.preview = tk.Label(display_frame, text="Upload an image or start your webcam",
                                bg="#0f172a", fg="#cbd5e1", font=("Arial", 15))
        self.preview.pack(fill="both", expand=True)

        footer = tk.Label(root, textvariable=self.status, anchor="w", padx=23, pady=12,
                          bg="#e2e8f0", fg="#334155", font=("Arial", 10))
        footer.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def load_model(self):
        if self.model is None:
            self.status.set("Loading YOLO model... first run may download weights.")
            self.root.update_idletasks()
            self.model = YOLO(MODEL_NAME)
        return self.model

    def detect(self, image_bgr):
        result = self.load_model().predict(source=image_bgr,
                                            conf=float(self.confidence.get()),
                                            verbose=False)[0]
        annotated = result.plot()
        names = result.names
        counts = Counter(names[int(i)] for i in result.boxes.cls.tolist())
        total = sum(counts.values())
        description = ", ".join(f"{name}: {count}" for name, count in counts.most_common(5))
        self.objects.set(f"Objects detected: {total}" + (f" | {description}" if description else ""))
        self.last_result = annotated.copy()
        self.save_button.config(state="normal")
        return annotated

    def show_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        width = max(self.preview.winfo_width() - 16, 500)
        height = max(self.preview.winfo_height() - 16, 350)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.config(image=self.photo, text="")

    def open_image(self):
        self.stop_webcam()
        path = filedialog.askopenfilename(title="Choose an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            # PIL supports paths containing spaces and Unicode characters.
            image_rgb = np.asarray(Image.open(path).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            annotated = self.detect(image_bgr)
            self.show_image(annotated)
            self.status.set(f"Analyzed image: {Path(path).name}")
        except Exception as exc:
            self.status.set("Could not analyze image.")
            messagebox.showerror("Detection error", str(exc))

    def start_webcam(self):
        if self.webcam_running:
            return
        try:
            self.load_model()
            # CAP_AVFOUNDATION is available on macOS, default works elsewhere.
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None
                raise RuntimeError("Could not open webcam. Check camera permissions or close other camera apps.")
            self.webcam_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.upload_button.config(state="disabled")
            self.status.set("Webcam running. Click Stop Webcam to finish.")
            self.next_frame()
        except Exception as exc:
            self.stop_webcam()
            messagebox.showerror("Webcam error", str(exc))

    def next_frame(self):
        if not self.webcam_running or self.camera is None:
            return
        success, frame = self.camera.read()
        if not success:
            self.stop_webcam()
            self.status.set("Webcam frame unavailable.")
            return
        try:
            annotated = self.detect(frame)
            self.show_image(annotated)
        except Exception as exc:
            self.stop_webcam()
            messagebox.showerror("Detection error", str(exc))
            return
        if self.webcam_running:
            self.root.after(20, self.next_frame)

    def stop_webcam(self):
        self.webcam_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.upload_button.config(state="normal")

    def save_result(self):
        if self.last_result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                   filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
                   initialfile="yolo_detection.jpg")
        if not path:
            return
        try:
            extension = Path(path).suffix.lower() or ".jpg"
            ok, buffer = cv2.imencode(extension, self.last_result)
            if not ok:
                raise RuntimeError("Image encoding failed")
            Path(path).write_bytes(buffer.tobytes())
            self.status.set(f"Saved result to {path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def close(self):
        self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()
