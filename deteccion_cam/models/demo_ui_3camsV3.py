# demo_ui_3cams.py (colocar dentro de /models)
import os, json, time, threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tensorflow as tf

# =================== CONFIG ===================
CAM_INDEXES    = [0, 2, 1]   # ajusta según tu PC
CAP_WIDTH      = 320
CAP_HEIGHT     = 240
SHOW_W, SHOW_H = 400, 300
ROI_RATIO      = 0.75        # recorte central para reducir ruido
INPUT_COL      = (16, 16)    # del entrenamiento de colores (meta)
INPUT_LET      = (32, 32)    # tu modelo de letras final
THRES_COL      = 0.65
THRES_LET      = 0.70
PRED_EVERY_MS  = 250
UI_REFRESH_MS  = 33
FONT_TITLE     = ("Segoe UI", 12, "bold")
FONT_TEXT      = ("Segoe UI", 10)

# Rutas (este archivo está dentro de /models)
MODELS_DIR = Path(__file__).resolve().parent
PATH_COL   = MODELS_DIR / "modelo_colores.keras"
PATH_LET   = MODELS_DIR / "modelo_letras.keras"
PATH_META  = MODELS_DIR / "modelo_meta.json"

# ================= UTILIDADES =================
def load_meta(path_meta: Path):
    defaults = {
        "classes_colors": ["Color_amarillo-","Color_azul-","Color_negro-","Color_plateado-","Color_rojo-","Color_verde-"],
        "classes_letters": ["Letra_h-","Letra_s-","Letra_u-"],
        "img_size": [16, 16]
    }
    if path_meta.exists():
        try:
            return json.loads(path_meta.read_text(encoding="utf-8"))
        except Exception:
            pass
    return defaults

def strip_prefix(name: str):
    # quita "Color_" / "Letra_" y guiones finales para mostrar bonito
    return name.replace("Color_", "").replace("Letra_", "").strip("-_ ")

def center_roi(frame, ratio=ROI_RATIO):
    if ratio >= 0.999:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)
    h, w = frame.shape[:2]
    rw, rh = int(w*ratio), int(h*ratio)
    x, y = (w - rw)//2, (h - rh)//2
    return frame[y:y+rh, x:x+rw], (x, y, rw, rh)

def preprocess_bgr_for_colors(bgr, size=INPUT_COL):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return img[None, ...]  # (1,H,W,3)

def preprocess_bgr_for_letters(bgr, size=INPUT_LET):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(-1, 0))  # (1,H,W,1)
    return img

# ================= CARGA MODELOS ==============
print("TensorFlow:", tf.__version__)
meta = load_meta(PATH_META)
labels_colors  = [strip_prefix(x) for x in meta.get("classes_colors", [])]
labels_letters = [strip_prefix(x) for x in meta.get("classes_letters", [])]

# (opcional) si quieres validar tamaños del meta:
# print("Meta img_size (colores):", meta.get("img_size"))

model_colors  = tf.keras.models.load_model(str(PATH_COL), compile=False)
model_letters = tf.keras.models.load_model(str(PATH_LET), compile=False)

# ============== ESTADO POR CÁMARA =============
class CamState:
    def __init__(self, cam_idx, name, video_label, signal_label, result_label):
        self.cam_idx       = cam_idx
        self.name          = name
        self.video_label   = video_label
        self.signal_label  = signal_label
        self.result_label  = result_label
        self.cap           = None
        self.running       = False
        self.frame_lock    = threading.Lock()
        self.latest_frame  = None
        self.latest_disp   = None
        self.pred_lock     = threading.Lock()
        self.pred_text     = "esperando…"
        self.fps_ema       = 0.0
        self.last_pred_ts  = 0.0

# ============== TKINTER UI ====================
root = tk.Tk()
root.title("Detector 3 Cámaras — Colores + Letras (TF)")
root.configure(bg="#1f2a7a")
for i in range(3):
    root.columnconfigure(i, weight=1)

def make_panel(col, cam_name):
    cont = tk.Frame(root, bg="#1f2a7a", padx=10, pady=10)
    cont.grid(row=0, column=col, sticky="n")

    video_frame = tk.Frame(cont, bg="#dfe6ff")
    video_frame.pack()
    video_label = tk.Label(video_frame, bg="#dfe6ff")
    video_label.pack()

    signal = tk.Label(cont, text="Señal del color-letra", bg="#eaf9ee", fg="#222",
                      font=FONT_TEXT, padx=14, pady=6)
    signal.pack(pady=(10,6))

    result = tk.Label(cont, text=f"{cam_name} • esperando…", bg="#ffffff", fg="#222",
                      width=28, height=2, font=FONT_TITLE)
    result.pack()
    return video_label, signal, result

panels = []
for i, idx in enumerate(CAM_INDEXES):
    cam_name = f"Cámara {i+1}"
    v, s, r = make_panel(i, cam_name)
    panels.append(CamState(idx, cam_name, v, s, r))

# ======== CAPTURA (hilo) ========
def capture_worker(state: CamState):
    cap = cv2.VideoCapture(state.cam_idx, cv2.CAP_DSHOW)  # DShow ayuda con latencia en Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state.cap = cap

    if not cap.isOpened():
        img = np.zeros((SHOW_H, SHOW_W, 3), dtype=np.uint8)
        cv2.putText(img, "Sin señal", (40, SHOW_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        state.latest_disp = img
        state.result_label.configure(text=f"{state.name} • sin cámara")
        return

    state.running = True
    t_last = time.time()
    while state.running:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt  = now - t_last
        t_last = now
        fps = 1.0/dt if dt > 0 else 0.0
        state.fps_ema = 0.9*state.fps_ema + 0.1*fps if state.fps_ema else fps

        with state.frame_lock:
            state.latest_frame = frame
            roi, (rx, ry, rw, rh) = center_roi(frame, ROI_RATIO)
            disp = frame.copy()
            cv2.rectangle(disp, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
            state.latest_disp = cv2.resize(disp, (SHOW_W, SHOW_H), interpolation=cv2.INTER_AREA)

    cap.release()

# ======== PREDICCIÓN (hilo) ========
def predict_worker(state: CamState):
    while True:
        if not state.running:
            time.sleep(0.05)
            continue
        now = time.time()
        if (now - state.last_pred_ts)*1000.0 < PRED_EVERY_MS:
            time.sleep(0.01)
            continue

        with state.frame_lock:
            frame = None if state.latest_frame is None else state.latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        roi, _ = center_roi(frame, ROI_RATIO)

        # --- dos preprocesos distintos ---
        x_col = preprocess_bgr_for_colors(roi, INPUT_COL)   # (1,16,16,3)
        x_let = preprocess_bgr_for_letters(roi, INPUT_LET)  # (1,32,32,1)

        # inferencias
        pc = model_colors.predict(x_col, verbose=0)[0]
        pl = model_letters.predict(x_let, verbose=0)[0]
        ic, il = int(np.argmax(pc)), int(np.argmax(pl))
        conf_c, conf_l = float(pc[ic]), float(pl[il])

        color_txt = f"{labels_colors[ic]} {conf_c:.2f}" if conf_c >= THRES_COL else "-"
        letra_txt = f"{labels_letters[il]} {conf_l:.2f}" if conf_l >= THRES_LET else "-"
        with state.pred_lock:
            state.pred_text = f"{state.name}\nColor: {color_txt}  |  Letra: {letra_txt}"
        state.last_pred_ts = now

        time.sleep(0.001)

# ======== UI REFRESH (hilo principal) ========
def refresh_ui():
    for st in panels:
        disp = st.latest_disp
        if disp is None:
            img = np.zeros((SHOW_H, SHOW_W, 3), dtype=np.uint8)
            cv2.putText(img, "Inicializando…", (20, SHOW_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            disp = img
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        st.video_label.imgtk = imgtk
        st.video_label.configure(image=imgtk)

        st.signal_label.configure(text=f"Señal del color-letra | FPS ~ {st.fps_ema:.1f}")
        with st.pred_lock:
            st.result_label.configure(text=st.pred_text)

    root.after(UI_REFRESH_MS, refresh_ui)

# ======== LANZAR HILOS Y UI ========
threads = []
for st in panels:
    t1 = threading.Thread(target=capture_worker, args=(st,), daemon=True)
    t1.start(); threads.append(t1)
    t2 = threading.Thread(target=predict_worker, args=(st,), daemon=True)
    t2.start(); threads.append(t2)

def on_close():
    for st in panels:
        st.running = False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(UI_REFRESH_MS, refresh_ui)
root.mainloop()
