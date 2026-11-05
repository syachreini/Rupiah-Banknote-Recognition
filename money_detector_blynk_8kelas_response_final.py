from pathlib import Path
from datetime import datetime
import time
import signal
import subprocess
import sys
import gc

import numpy as np
from PIL import Image
from gpiozero import Button
from picamera2 import Picamera2
from libcamera import controls
import tensorflow as tf

try:
    import requests
except Exception:
    requests = None


# ==================================================
# MONEY DETECTOR TUNARA + BLYNK
# ==================================================
# Fitur:
# - Meniru kode capture dataset:
#   Picamera2, create_still_configuration, 1640x1232, AWB Daylight, capture_file()
# - Kamera hanya ON saat tombol CAPTURE ditekan
# - Setelah capture: stop(), close(), del, gc.collect(), jeda 2 detik
# - Jika capture gagal, program tidak langsung mati
# - Audio pakai pw-play
# - Blynk update:
#   V0 = Nominal Terakhir
#   V1 = Confidence (%)
#   V2 = Status Alat
#   V3 = Jumlah Scan
#   V4 = Detail Prediksi
# ==================================================


# ==================================================
# PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mobilenetv2_uang_rupiah_8kelas.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"
AUDIO_DIR = BASE_DIR / "audio"

DEBUG_RAW_PATH = BASE_DIR / "debug_last_capture.jpg"
CAPTURE_LOG_DIR = BASE_DIR / "captures"


# ==================================================
# GPIO BUTTON
# ==================================================
# Tombol ON/OFF:
# pin fisik 13 = GPIO27
SHUTDOWN_BUTTON_GPIO = 27
SHUTDOWN_HOLD_SECONDS = 2

# Tombol CAPTURE:
# pin fisik 18 = GPIO24
CAPTURE_BUTTON_GPIO = 24

# Untuk alat portable final: True.
# Kalau mau tes tanpa mematikan Raspberry Pi, ubah sementara menjadi False.
ENABLE_REAL_SHUTDOWN = True


# ==================================================
# AUDIO
# ==================================================
AUDIO_PLAYER = "pw-play"


# ==================================================
# BLYNK IOT
# ==================================================
BLYNK_ENABLED = True
BLYNK_AUTH_TOKEN = "_1ZhMBllLCVG6azsGv8Z9aKUHn2jAZzs"
BLYNK_SERVER = "https://blynk.cloud"

# Virtual Pin:
# V0 = Nominal Terakhir
# V1 = Confidence
# V2 = Status Alat
# V3 = Jumlah Scan
# V4 = Detail Prediksi
# V9 = Response Time (detik)
scan_count = 0
response_time = 0


# ==================================================
# KONFIGURASI MODEL / LOGIKA
# ==================================================
CONFIDENCE_THRESHOLD = 0.85
DELAY_AFTER_RESULT_SECONDS = 1.5
PLAY_READY_AFTER_RESULT = True


# ==================================================
# KONFIGURASI KAMERA - SAMA DENGAN KODE DATASET
# ==================================================
WIDTH = 1640
HEIGHT = 1232

AWB_MODE = controls.AwbModeEnum.Daylight
CAMERA_WARMUP_SECONDS = 2.0

# Jeda setelah kamera ditutup agar state kembali Available.
CAMERA_RELEASE_DELAY_SECONDS = 2.0

# Jika pembukaan kamera gagal, coba ulang beberapa kali.
CAMERA_OPEN_RETRY = 2
CAMERA_OPEN_RETRY_DELAY_SECONDS = 2.0

SAVE_ALL_CAPTURES = True


# ==================================================
# GLOBAL STATE
# ==================================================
running = True
retry_mode = False
previous_low_conf_label = None
shutdown_requested = False
error_happened = False


# ==================================================
# SIGNAL HANDLER
# ==================================================
def stop_program(signum=None, frame=None):
    global running
    running = False


signal.signal(signal.SIGTERM, stop_program)
signal.signal(signal.SIGINT, stop_program)


# ==================================================
# BLYNK UPDATE
# ==================================================
def blynk_update(**datastreams):
    """
    Kirim data ke Blynk Cloud.

    Catatan penting:
    Endpoint /external/api/update kadang hanya memperbarui satu virtual pin
    dengan stabil. Karena itu, setiap pin dikirim satu per satu agar V0, V1,
    V2, V3, dan V4 semuanya benar-benar berubah di Blynk.
    """
    if not BLYNK_ENABLED:
        return

    if requests is None:
        print("[BLYNK WARNING] Library requests belum tersedia. Jalankan: pip install requests", flush=True)
        return

    if not BLYNK_AUTH_TOKEN:
        print("[BLYNK WARNING] Token Blynk kosong.", flush=True)
        return

    url = f"{BLYNK_SERVER}/external/api/update"

    for pin, value in datastreams.items():
        params = {
            "token": BLYNK_AUTH_TOKEN,
            pin: value,
        }

        try:
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                print(f"[BLYNK] {pin} update berhasil: {value}", flush=True)
            else:
                print(f"[BLYNK WARNING] {pin} gagal: {response.status_code} {response.text}", flush=True)

        except Exception as e:
            # Jangan sampai error internet membuat alat berhenti
            print(f"[BLYNK WARNING] Tidak bisa update {pin}: {e}", flush=True)


# ==================================================
# AUDIO
# ==================================================
def play_audio(filename):
    path = AUDIO_DIR / filename

    if not path.exists():
        print(f"[WARNING] File audio tidak ditemukan: {path}", flush=True)
        return

    print(f"[AUDIO] {filename}", flush=True)
    subprocess.run([AUDIO_PLAYER, str(path)], check=False)


# ==================================================
# TOMBOL
# ==================================================
def request_shutdown():
    global running
    global shutdown_requested

    if shutdown_requested:
        return

    print("[BUTTON] Tombol ON/OFF ditahan. Sistem akan ditutup.", flush=True)
    shutdown_requested = True
    running = False


shutdown_button = Button(
    SHUTDOWN_BUTTON_GPIO,
    pull_up=True,
    bounce_time=0.1,
    hold_time=SHUTDOWN_HOLD_SECONDS,
)

capture_button = Button(
    CAPTURE_BUTTON_GPIO,
    pull_up=True,
    bounce_time=0.3,
)

shutdown_button.when_held = request_shutdown


# ==================================================
# CEK FILE WAJIB
# ==================================================
required_files = [
    MODEL_PATH,
    LABELS_PATH,
    AUDIO_DIR / "aktif.wav",
    AUDIO_DIR / "silahkan.wav",
    AUDIO_DIR / "ulang.wav",
    AUDIO_DIR / "nonaktif.wav",
    AUDIO_DIR / "kosong.wav",
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")


# ==================================================
# LOAD LABELS
# ==================================================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

print("Labels:", labels, flush=True)

for label in labels:
    audio_file = AUDIO_DIR / f"{label}.wav"
    if not audio_file.exists():
        print(f"[WARNING] Audio nominal tidak ditemukan: {audio_file}", flush=True)


# ==================================================
# LOAD MODEL TFLITE
# ==================================================
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]

input_shape = input_details[0]["shape"]
output_shape = output_details[0]["shape"]

IMG_HEIGHT = int(input_shape[1])
IMG_WIDTH = int(input_shape[2])

print("\nInput detail:", flush=True)
print(input_details[0], flush=True)

print("\nOutput detail:", flush=True)
print(output_details[0], flush=True)

print(f"\nUkuran input model: {IMG_WIDTH}x{IMG_HEIGHT}", flush=True)

try:
    output_classes = int(output_shape[-1])
    if output_classes != len(labels):
        print(
            f"[WARNING] Jumlah output model ({output_classes}) tidak sama dengan jumlah label ({len(labels)}).",
            flush=True,
        )
except Exception:
    pass


# ==================================================
# CAMERA CAPTURE - SAMA DENGAN KODE DATASET + RELEASE FIX
# ==================================================
def cleanup_camera(picam2):
    if picam2 is None:
        return

    try:
        picam2.stop()
    except Exception as e:
        print(f"[WARNING] Gagal stop kamera: {e}", flush=True)

    try:
        picam2.close()
    except Exception as e:
        print(f"[WARNING] Gagal close kamera: {e}", flush=True)

    del picam2
    gc.collect()

    print("Kamera dimatikan dan dilepas.", flush=True)
    time.sleep(CAMERA_RELEASE_DELAY_SECONDS)


def capture_once():
    """
    Kamera hanya aktif saat tombol CAPTURE ditekan,
    dengan setting yang sama seperti kode capture dataset.
    """
    last_error = None

    for attempt in range(1, CAMERA_OPEN_RETRY + 2):
        picam2 = None

        try:
            print(f"Menyalakan kamera... percobaan {attempt}", flush=True)

            picam2 = Picamera2()

            config = picam2.create_still_configuration(
                main={"size": (WIDTH, HEIGHT)}
            )

            picam2.configure(config)

            picam2.set_controls({
                "AwbEnable": True,
                "AwbMode": AWB_MODE,
            })

            picam2.start()

            print(f"Menunggu kamera stabil {CAMERA_WARMUP_SECONDS} detik...", flush=True)
            time.sleep(CAMERA_WARMUP_SECONDS)

            print(f"Mengambil gambar: {DEBUG_RAW_PATH}", flush=True)
            picam2.capture_file(str(DEBUG_RAW_PATH))

            if not DEBUG_RAW_PATH.exists():
                raise FileNotFoundError(f"Gambar capture tidak ditemukan: {DEBUG_RAW_PATH}")

            print(f"Gambar berhasil disimpan: {DEBUG_RAW_PATH}", flush=True)

            image = Image.open(DEBUG_RAW_PATH).convert("RGB")
            frame_rgb = np.array(image)

            return frame_rgb

        except Exception as e:
            last_error = e
            print(f"[WARNING] Capture gagal pada percobaan {attempt}: {e}", flush=True)

            if attempt <= CAMERA_OPEN_RETRY:
                print(
                    f"Menunggu {CAMERA_OPEN_RETRY_DELAY_SECONDS} detik lalu mencoba kamera lagi...",
                    flush=True,
                )
                time.sleep(CAMERA_OPEN_RETRY_DELAY_SECONDS)

        finally:
            cleanup_camera(picam2)

    raise RuntimeError(f"Capture gagal setelah beberapa percobaan: {last_error}")


# ==================================================
# PREPROCESS
# ==================================================
def preprocess_frame(frame_rgb):
    img = Image.fromarray(frame_rgb).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    arr = np.array(img)

    # Sesuai training:
    # preprocess_input sudah masuk ke graph model,
    # jadi input ke TFLite cukup RGB 0-255.
    if input_dtype == np.float32:
        arr = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(input_dtype)

    arr = np.expand_dims(arr, axis=0)
    return arr


# ==================================================
# PREDICT
# ==================================================
def predict_from_frame(frame_rgb):
    input_data = preprocess_frame(frame_rgb)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_index)[0]

    if output_dtype in [np.uint8, np.int8]:
        scale, zero_point = output_details[0]["quantization"]
        output_data = scale * (output_data.astype(np.float32) - zero_point)

    pred_idx = int(np.argmax(output_data))
    pred_label = labels[pred_idx]
    confidence = float(output_data[pred_idx])

    return pred_label, confidence, output_data


# ==================================================
# SIMPAN CAPTURE UNTUK DEBUG / TRAINING ULANG
# ==================================================
def save_capture_log(pred_label, confidence):
    if not SAVE_ALL_CAPTURES:
        return

    if not DEBUG_RAW_PATH.exists():
        return

    CAPTURE_LOG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_conf = f"{confidence:.2f}".replace(".", "p")

    dst_path = CAPTURE_LOG_DIR / f"{timestamp}_pred_{pred_label}_conf_{safe_conf}.jpg"

    Image.open(DEBUG_RAW_PATH).convert("RGB").save(dst_path)
    print(f"Capture log: {dst_path}", flush=True)


def make_probability_text(probs, pred_label, confidence):
    """
    Ringkasan probabilitas untuk V4 Detail Prediksi.
    Format dibuat Top 3 agar sesuai dengan nominal terakhir dan mudah dibaca di Blynk.
    """
    top_indices = np.argsort(probs)[::-1][:3]

    lines = [
        f"Hasil Rp{pred_label} {confidence * 100:.2f}%"
    ]

    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx]
        percent = float(probs[idx]) * 100
        lines.append(f"Top {rank} Rp{label} {percent:.2f}%")

    return " | ".join(lines)


# ==================================================
# LOGIKA ALAT
# ==================================================
def handle_prediction(frame_rgb):
    global retry_mode
    global response_time
    global previous_low_conf_label
    global scan_count

    start_response = time.perf_counter()

    pred_label, confidence, probs = predict_from_frame(frame_rgb)

    response_time = round(
        time.perf_counter() - start_response,
        3
    )

    print(f"Response Time: {response_time} detik", flush=True)
    save_capture_log(pred_label, confidence)

    scan_count += 1
    confidence_percent = round(confidence * 100, 2)
    prob_detail = make_probability_text(probs, pred_label, confidence)

    print("\n====================================", flush=True)
    print(f"Prediksi   : Rp{pred_label}", flush=True)
    print(f"Confidence : {confidence:.4f}", flush=True)
    print("------------------------------------", flush=True)

    for label, prob in zip(labels, probs):
        print(f"Rp{label}: {float(prob):.4f}", flush=True)

    print("====================================", flush=True)

    # Kirim hasil prediksi ke Blynk.
    if confidence >= CONFIDENCE_THRESHOLD:
        status_text = "Prediksi berhasil"
    else:
        status_text = "Confidence rendah - pindai ulang"

    print("[BLYNK PAYLOAD]", flush=True)
    print("V0:", f"Rp{pred_label}", flush=True)
    print("V1:", confidence_percent, flush=True)
    print("V2:", status_text, flush=True)
    print("V3:", scan_count, flush=True)
    print("V4:", prob_detail, flush=True)

    blynk_update(
        v0=f"Rp{pred_label}",
        v1=confidence_percent,
        v2=status_text,
        v3=scan_count,
        v4=prob_detail,
        v9=response_time
    )


    # Kasus kelas kosong (tidak ada objek uang)
    if pred_label.lower() == "kosong":
        print("Keputusan: area scan kosong.", flush=True)

        blynk_update(
            v0="Kosong",
            v1=confidence_percent,
            v2="Tidak ada uang",
            v3=scan_count,
            v4=prob_detail,
            v9=response_time
        )

        play_audio("kosong.wav")

        retry_mode = False
        previous_low_conf_label = None

        return

    # Kasus 1: confidence tinggi
    if confidence >= CONFIDENCE_THRESHOLD:
        print("Keputusan: confidence cukup, nominal dibacakan.", flush=True)
        play_audio(f"{pred_label}.wav")

        retry_mode = False
        previous_low_conf_label = None

        time.sleep(DELAY_AFTER_RESULT_SECONDS)

        if PLAY_READY_AFTER_RESULT and running:
            blynk_update(v2="Siap Scan")
            play_audio("silahkan.wav")

        return

    # Kasus 2: confidence rendah pertama kali
    if not retry_mode:
        print("Keputusan: confidence rendah pertama kali, minta pindai ulang.", flush=True)
        play_audio("ulang.wav")

        retry_mode = True
        previous_low_conf_label = pred_label
        return

    # Kasus 3: confidence rendah kedua kali, tapi prediksi sama
    if retry_mode and pred_label == previous_low_conf_label:
        print("Keputusan: confidence rendah, tapi prediksi sama dua kali. Nominal dibacakan.", flush=True)
        blynk_update(v2="Prediksi sama 2x - nominal dibacakan")
        play_audio(f"{pred_label}.wav")

        retry_mode = False
        previous_low_conf_label = None

        time.sleep(DELAY_AFTER_RESULT_SECONDS)

        if PLAY_READY_AFTER_RESULT and running:
            blynk_update(v2="Siap Scan")
            play_audio("silahkan.wav")

        return

    # Kasus 4: confidence rendah dan prediksi berubah
    print("Keputusan: confidence rendah dan prediksi berubah, minta pindai ulang lagi.", flush=True)
    blynk_update(v2="Confidence rendah - prediksi berubah")
    play_audio("ulang.wav")

    retry_mode = True
    previous_low_conf_label = pred_label


# ==================================================
# MAIN
# ==================================================
def main():
    global error_happened
    global scan_count

    print("\n====================================", flush=True)
    print(" MONEY DETECTOR TUNARA + BLYNK", flush=True)
    print("====================================", flush=True)
    print(f"Threshold confidence : {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"Shutdown button      : GPIO{SHUTDOWN_BUTTON_GPIO}", flush=True)
    print(f"Capture button       : GPIO{CAPTURE_BUTTON_GPIO}", flush=True)
    print(f"Audio player         : {AUDIO_PLAYER}", flush=True)
    print(f"Camera size          : {WIDTH}x{HEIGHT}", flush=True)
    print(f"White balance        : Daylight", flush=True)
    print(f"Camera release delay : {CAMERA_RELEASE_DELAY_SECONDS}s", flush=True)
    print(f"Blynk enabled        : {BLYNK_ENABLED}", flush=True)
    print("Mode: kamera aktif hanya saat tombol capture", flush=True)
    print("====================================\n", flush=True)

    try:
        blynk_update(
            v0="-",
            v1=0,
            v2="Alat aktif",
            v3=scan_count,
            v4="-",
            v9=0
        )

        play_audio("aktif.wav")
        time.sleep(0.5)

        play_audio("silahkan.wav")
        blynk_update(v2="Siap Scan")

        print("Program siap.", flush=True)
        print("Kamera OFF saat menunggu.", flush=True)
        print("Letakkan uang di area scan, lepaskan tangan, lalu tekan tombol CAPTURE.", flush=True)
        print("Tahan tombol ON/OFF selama 2 detik untuk menutup sistem.", flush=True)

        while running:
            if capture_button.is_pressed:
                print("[BUTTON] Tombol CAPTURE ditekan.", flush=True)

                blynk_update(v2="Memproses scan")

                try:
                    frame_rgb = capture_once()
                    handle_prediction(frame_rgb)
                except Exception as e:
                    # Capture error tidak boleh langsung mematikan alat.
                    print(f"[ERROR] Capture/prediksi gagal: {e}", flush=True)
                    blynk_update(v2=f"Error scan: {str(e)[:80]}")
                    play_audio("ulang.wav")

                capture_button.wait_for_release()
                time.sleep(0.5)

            time.sleep(0.05)

    except Exception as e:
        error_happened = True
        print(f"[ERROR] Error utama program: {e}", flush=True)
        blynk_update(v2=f"Error utama: {str(e)[:80]}")

    finally:
        print("Menutup sistem...", flush=True)

        blynk_update(v2="Alat nonaktif / shutdown")

        play_audio("nonaktif.wav")
        print("Sistem nonaktif.", flush=True)

        if shutdown_requested and ENABLE_REAL_SHUTDOWN:
            time.sleep(0.5)
            print("Mematikan Raspberry Pi...", flush=True)
            subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"], check=False)

        if error_happened:
            sys.exit(1)


if __name__ == "__main__":
    main()
