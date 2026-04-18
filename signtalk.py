# app.py – SignTalk (Real-time Translation with Gramformer)

import os, sys, pathlib, time, cv2, torch, mediapipe as mp, joblib
import numpy as np
import streamlit as st
from collections import deque, Counter
from threading import Thread

# ===============================================================
# Fix YOLOv5 pathlib issue on Windows
# ===============================================================
try:
    if os.name == 'nt':
        sys.modules["pathlib._local"] = pathlib
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
except Exception as e:
    print("Pathlib patch warning:", e)

# ===============================================================
# Streamlit page setup
# ===============================================================
st.set_page_config(page_title="SignTalk", page_icon="🤟", layout="wide")
st.title("🤟 SignTalk: AI Sign Language Translator")
st.markdown("**Live translation** of sign language into grammatically correct sentences using YOLOv5, Mediapipe, SVM, and NLP integration.")

# ===============================================================
# Configuration
# ===============================================================
REPO_PATH = r"C:/Users/PRIYANGA/yolov5"
WEIGHTS_PATH = r"C:/Users/PRIYANGA/yolov5/runs/train/exp5/weights/best.pt"
SVM_MODEL_FILE = "svm_mediapipe.pkl"

USE_GRAMFORMER = True     # ✅ Enable grammar correction
YOLO_CONF = 0.15
BUFFER_LEN = 5
REQUIRED_COUNT = 3
PAUSE_SEC = 2.0

# ===============================================================
# Load models (cached)
# ===============================================================
@st.cache_resource
def load_all_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load(REPO_PATH, 'custom', path=WEIGHTS_PATH,
                           source='local', force_reload=False)
    model.to(device).eval()

    bundle = joblib.load(SVM_MODEL_FILE)
    scaler, svm = bundle["scaler"], bundle["svm"]

    gf = None
    if USE_GRAMFORMER:
        try:
            from gramformer import Gramformer
            gf = Gramformer(models=1)
        except Exception as e:
            st.warning(f"Gramformer failed to load: {e}")
            gf = None
    return model, scaler, svm, device, gf

model, scaler, svm, device, gf = load_all_models()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ===============================================================
# Helper functions
# ===============================================================
def run_yolo(rgb):
    try:
        results = model(rgb)
        if results.xyxy[0].shape[0] > 0:
            best = max(results.xyxy[0].cpu().numpy(), key=lambda r: float(r[4]))
            conf = float(best[4])
            cls = int(best[5])
            label = model.names[cls]
            if conf >= YOLO_CONF:
                return label, conf
    except Exception as e:
        print("YOLO error:", e)
    return None, 0.0

def run_svm(landmarks):
    try:
        X = np.array(landmarks).reshape(1, -1)
        Xs = scaler.transform(X)
        probs = svm.predict_proba(Xs)[0]
        idx = np.argmax(probs)
        return svm.classes_[idx], float(probs[idx])
    except Exception:
        return None, 0.0

def expert_decision(y_label, y_conf, s_label, s_conf):
    if s_label is None and y_label: return y_label
    if y_label is None and s_label: return s_label
    if y_label == s_label: return y_label
    return y_label if y_conf > s_conf else s_label

def correct_sentence_async(sentence, gf, result_container):
    """Runs Gramformer in a separate thread so it doesn't block video."""
    try:
        corrected = gf.correct(sentence)
        if isinstance(corrected, list):
            result_container["sentence"] = corrected[0]
    except Exception:
        result_container["sentence"] = sentence

# ===============================================================
# Real-time live stream
# ===============================================================
start = st.button("🎥 Start Live Translation")
FRAME = st.empty()
OUTPUT = st.empty()

if start:
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=BUFFER_LEN)
    tokens = []
    last_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    st.info("Press **Stop** or close the tab to end session.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO + Mediapipe + SVM
        y_label, y_conf = run_yolo(rgb)
        s_label, s_conf = None, 0.0
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                coords = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
                s_label, s_conf = run_svm(coords)

        final_label = expert_decision(y_label, y_conf, s_label, s_conf)
        buffer.append(final_label)

        if len(buffer) == BUFFER_LEN:
            c = Counter(buffer)
            common, count = c.most_common(1)[0]
            if common and count >= REQUIRED_COUNT:
                if not tokens or tokens[-1] != common:
                    tokens.append(common)
                    last_time = time.time()
            buffer.clear()

        # Process sentence after pause
        if tokens and (time.time() - last_time > PAUSE_SEC):
            sentence = " ".join(tokens)
            corrected_sentence = sentence
            if gf:
                result_container = {"sentence": sentence}
                thread = Thread(target=correct_sentence_async,
                                args=(sentence, gf, result_container))
                thread.start()
                thread.join(timeout=3)
                corrected_sentence = result_container["sentence"]
            OUTPUT.markdown(f"### 🗣️ {corrected_sentence}")
            tokens.clear()

        if final_label:
            cv2.putText(frame, f"{final_label}", (30, 50),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        FRAME.image(frame, channels="BGR")

    cap.release()
    hands.close()
    st.success("Session ended.")
