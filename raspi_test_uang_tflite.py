import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = Path(__file__).resolve().parent / "model_uang_mobilenetv2_best.tflite"
LABEL_PATH = Path(__file__).resolve().parent / "class_names.json"
AUDIO_DIR = Path(__file__).resolve().parent / "audio"

CONFIDENCE_THRESHOLD = 0.70
AUDIO_DEVICE = "plughw:2,0"

# Ukuran preview kamera
FRAME_W = 640
FRAME_H = 480

# ROI kotak tengah
ROI_W = 400
ROI_H = 200

# Warna tampilan (BGR)
COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (0, 255, 255)
COLOR_STATUS = (255, 255, 255)

AUDIO_MAP = {
    "1000": AUDIO_DIR / "1000.wav",
    "2000": AUDIO_DIR / "2000.wav",
    "5000": AUDIO_DIR / "5000.wav",
    "10000": AUDIO_DIR / "10000.wav",
    "20000": AUDIO_DIR / "20000.wav",
    "50000": AUDIO_DIR / "50000.wav",
    "100000": AUDIO_DIR / "100000.wav",
    "ulang": AUDIO_DIR / "ulang.wav",
    "aktif": AUDIO_DIR / "aktif.wav",
    "nonaktif": AUDIO_DIR / "nonaktif.wav",
    "scan": AUDIO_DIR / "scan.wav",
    "silahkan": AUDIO_DIR / "silahkan.wav",
}


def play_audio(label: str, blocking: bool = True) -> None:
    audio_path = AUDIO_MAP.get(label)
    if not audio_path or not audio_path.exists():
        print(f"Audio tidak ditemukan untuk label: {label}")
        return

    cmd = ["aplay", "-D", AUDIO_DEVICE, str(audio_path)]
    if blocking:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def load_labels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        try:
            labels = [labels[str(i)] for i in range(len(labels))]
        except Exception:
            labels = list(labels.values())

    return labels


def setup_interpreter(model_path: Path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_image(image_rgb: np.ndarray, input_details):
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    h = int(input_shape[1])
    w = int(input_shape[2])

    resized = cv2.resize(image_rgb, (w, h))
    input_data = np.expand_dims(resized, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(input_dtype)

    return input_data


def run_inference(interpreter, input_details, output_details, image_rgb: np.ndarray):
    input_data = preprocess_image(image_rgb, input_details)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    if output_details[0]["dtype"] != np.float32:
        scale, zero_point = output_details[0]["quantization"]
        if scale > 0:
            output = scale * (output.astype(np.float32) - zero_point)
        else:
            output = output.astype(np.float32)

    pred_idx = int(np.argmax(output))
    confidence = float(output[pred_idx])

    if confidence > 1.0 or np.min(output) < 0:
        exp_x = np.exp(output - np.max(output))
        probs = exp_x / np.sum(exp_x)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    return pred_idx, confidence, output


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {LABEL_PATH}")

    class_names = load_labels(LABEL_PATH)
    interpreter, input_details, output_details = setup_interpreter(MODEL_PATH)

    print("Class names:", class_names)
    print("Model loaded successfully.")
    print(f"Menggunakan output audio: {AUDIO_DEVICE}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    x1 = (FRAME_W - ROI_W) // 2
    y1 = (FRAME_H - ROI_H) // 2
    x2 = x1 + ROI_W
    y2 = y1 + ROI_H

    last_result = "Belum ada scan"
    last_conf = 0.0
    last_spoken_time = 0.0

    if AUDIO_MAP["aktif"].exists():
        play_audio("aktif", blocking=False)
    elif AUDIO_MAP["silahkan"].exists():
        play_audio("silahkan", blocking=False)

    print("Tekan SPACE untuk scan, Q untuk keluar.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), COLOR_BOX, 2)

            cv2.putText(
                frame_bgr,
                "Arahkan uang ke kotak hijau",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_STATUS,
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Hasil: {last_result}",
                (20, FRAME_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Confidence: {last_conf:.2f}",
                (20, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
            )

            cv2.imshow("Deteksi Uang Tunanetra", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if AUDIO_MAP["scan"].exists():
                    play_audio("scan", blocking=False)

                roi_rgb = frame_rgb[y1:y2, x1:x2]
                pred_idx, confidence, _ = run_inference(
                    interpreter, input_details, output_details, roi_rgb
                )

                if 0 <= pred_idx < len(class_names):
                    pred_label = str(class_names[pred_idx])
                else:
                    pred_label = "tidak diketahui"

                if confidence >= CONFIDENCE_THRESHOLD and pred_label in AUDIO_MAP:
                    last_result = f"Rp {pred_label}"
                    last_conf = confidence
                    print(f"Prediksi: {last_result} | confidence={confidence:.4f}")

                    now = time.time()
                    if now - last_spoken_time > 1.0:
                        play_audio(pred_label)
                        last_spoken_time = now
                else:
                    last_result = "Tidak yakin, ulangi"
                    last_conf = confidence
                    print(f"Prediksi tidak yakin | confidence={confidence:.4f}")
                    play_audio("ulang")

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        if AUDIO_MAP["nonaktif"].exists():
            play_audio("nonaktif", blocking=False)


if __name__ == "__main__":
    main()
