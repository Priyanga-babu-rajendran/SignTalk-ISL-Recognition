import os, sys, pathlib, time, cv2, torch, mediapipe as mp, joblib
import numpy as np
import streamlit as st
from collections import deque, Counter
from threading import Thread

import zipfile

def extract_if_needed():
    try:
        if not os.path.exists("best.pt") and os.path.exists("best.zip"):
            with zipfile.ZipFile("best.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            print("✅ Extracted best.pt")

        if not os.path.exists("svm_mediapipe.pkl") and os.path.exists("svm_mediapipe.zip"):
            with zipfile.ZipFile("svm_mediapipe.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            print("✅ Extracted svm model")

    except Exception as e:
        print("❌ Extraction failed:", e)
            
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
st.markdown("Live translation of sign language into grammatically correct sentences using YOLOv5, Mediapipe, SVM, and NLP integration.")

# ===============================================================
# Configuration
# ===============================================================

REPO_PATH = "ultralytics/yolov5"
WEIGHTS_PATH = "best.pt"
SVM_MODEL_FILE = "svm_mediapipe.pkl"

USE_GRAMFORMER = True    # ✅ Enable grammar correction
YOLO_CONF = 0.15
BUFFER_LEN = 7
REQUIRED_COUNT = 5
PAUSE_SEC = 2.0  

# ===============================================================
# Load models (cached)
# ===============================================================

@st.cache_resource
def load_all_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Check if files exist BEFORE loading ---
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"YOLO Weights Path NOT FOUND: {WEIGHTS_PATH}")
        return None, None, None, None, None
    if not os.path.exists(SVM_MODEL_FILE):
        st.error(f"SVM Model NOT FOUND: {SVM_MODEL_FILE}")
        return None, None, None, None, None
    
    st.info(f"Loading models... (Device: {device})")
    
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=WEIGHTS_PATH, force_reload=False)
    model.to(device).eval()

    bundle = joblib.load(SVM_MODEL_FILE)
    scaler, svm = bundle["scaler"], bundle["svm"]

    gf = None
    if USE_GRAMFORMER:
        try:
            from gramformer import Gramformer
            # This line downloads the model if not present
            gf = Gramformer(models=1) 
            st.write("✅ Gramformer model loaded successfully")
        except Exception as e:
            st.warning(f"Gramformer failed to load: {e}")
            gf = None
            
    return model, scaler, svm, device, gf

extract_if_needed()

with st.spinner("Loading AI models, please wait..."):
    model, scaler, svm, device, gf = load_all_models()

# Stop the app if models failed to load
if model is None or scaler is None or svm is None:
    st.error("A critical model failed to load. The app cannot start.")
    st.stop()
else:
    if gf:
        st.success("✅ All AI Models (YOLO, SVM, Gramformer) loaded.")
    else:
        st.success("✅ AI Models (YOLO, SVM) loaded. Gramformer disabled.")


try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

except Exception as e:
    st.error(f"Mediapipe failed to initialize: {e}")
    st.stop()
    
# mp_hands = mp.solutions.hands if hasattr(mp, "solutions") else None
# mp_drawing = mp.solutions.drawing_utils if hasattr(mp, "solutions") else None

# if mp_hands is None:
#     st.error("Mediapipe not properly installed")
#     st.stop()
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

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
    
def expert_decision(y_label, y_conf, s_label, s_conf, yolo_thresh=0.2, svm_thresh=0.4): 
    """
    Expert logic: Combine YOLO and SVM.
    Only accept detections above a threshold.
    """
    if y_label and y_conf < yolo_thresh:
        y_label = None
    if s_label and s_conf < svm_thresh:
        s_label = None

    # If both are None, ignore frame
    if not y_label and not s_label: return None

    # If one is missing, use the other
    if not y_label: return s_label
    if not s_label: return y_label

    # If both agree, accept
    if y_label == s_label: return y_label
    
    # If they disagree, pick the one with higher confidence
    return y_label if y_conf > s_conf else s_label


# --- THIS IS THE NEW THREAD FUNCTION ---
def correct_sentence_in_background(sentence, gf_model, job_container):
    """Run grammar correction with preprocessing and fallback handling."""
    
    clean_sentence = sentence # Default fallback
    corrected_sentence = sentence # Default fallback
    
    try:
        token_map = {
            # Words
            "eye": "eye",
            "join": "went",
            "doctor": "doctor",
            "eat": "eat",
            "thank": "thank you",
            "book": "book",
            "good": "good",
            "i": "I",
            "me": "me",
            "my": "my",
            "you": "you",
            "your": "your",
            "go": "go",
            "home": "home",
            "work": "work",
            "what": "what",
            "where": "where",
            "when": "when",
            "why": "why",
            "how": "how",
            
            # --- Words that need verbs ---
            "quiet": "I am quiet",
            "sleep": "I am sleeping",
            "hungry": "I am hungry",
            "happy": "I am happy",
            "sad": "I am sad",
        }
        
        # This maps "I quiet" to "I am quiet"
        tokens = [token_map.get(w.lower(), w) for w in sentence.split()]
        
        # This maps "i" to "I"
        tokens = ["I" if w.lower() == "i" else w for w in tokens]
        
        # If "I" is in the sentence, move it to the front to help Gramformer
        if "I" in tokens and tokens[0] != "I":
            tokens.remove("I")
            tokens.insert(0, "I")

        clean_sentence = " ".join(tokens).strip().capitalize()
        
        if not clean_sentence.endswith('.'):
            clean_sentence += '.'

        print(f"Gramformer: Attempting to correct: '{clean_sentence}'")
        
        # Run the correction
        corrected = gf_model.correct(clean_sentence, max_candidates=1)
        
        if isinstance(corrected, list) and corrected:
            corrected_sentence = corrected[0]
        elif isinstance(corrected, str):
            corrected_sentence = corrected
        else:
            corrected_sentence = clean_sentence # Fallback
            
    except Exception as e:
        print(f"--- GRAMFORMER CORRECTION FAILED ---")
        print(f"Input sentence was: '{clean_sentence}'")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        corrected_sentence = clean_sentence # Fallback to cleaned sentence
    finally:
        # Put the result in the shared dictionary
        print(f"Gramformer: Setting result: '{corrected_sentence}'")
        job_container["result"] = corrected_sentence
        
# ===============================================================
# Real-time live stream
# ===============================================================

start = st.button("🎥 Start Live Translation")
FRAME = st.empty()
OUTPUT = st.empty()

# ---  Shared dictionary to communicate between threads ---
correction_job = {"thread": None, "result": None, "sentence": None}

if start:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Please close any other app using it and restart.")
        st.stop()
    
    buffer = deque(maxlen=BUFFER_LEN)
    tokens = []
    last_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    st.info("Webcam opened. Press Stop or close the tab to end session.")

    while cap.isOpened():
        
        # ---  Check for finished jobs at the START of the loop ---
        if correction_job["result"] is not None:
            # A thread has finished. Update the UI from the MAIN thread.
            print(f"MAIN LOOP: Detected result: '{correction_job['result']}'")
            OUTPUT.markdown(f"### 🗣 {correction_job['result']}")
            # Reset the job
            correction_job = {"thread": None, "result": None, "sentence": None}

        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam frame failed. Stopping.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Run Models (YOLO + SVM)
        y_label, y_conf = run_yolo(rgb)
        s_label, s_conf = None, 0.0
        
        # 3. Run Mediapipe
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                coords = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
                s_label, s_conf = run_svm(coords)

        # 4. Get Final Decision
        final_label = expert_decision(y_label, y_conf, s_label, s_conf)
        
        # 5. Buffer Logic
        # We only care about valid, non-None detections
        if final_label:
            buffer.append(final_label)
            
            # Check if buffer is full
            if len(buffer) == BUFFER_LEN:
                c = Counter(buffer)
                common, count = c.most_common(1)[0]
                
                # Check if detection is stable
                if common and count >= REQUIRED_COUNT:
                    if not tokens or tokens[-1] != common:
                        tokens.append(common)
                        last_time = time.time() # Reset pause timer
                        print(f"App: Detected new token: '{common}'. Current tokens: {tokens}")
                buffer.clear()

        # 6. Process Sentence after Pause
        # Check if a correction isn't already running AND
        # if there are tokens AND
        # if enough time has passed since the last new sign
        if (correction_job["thread"] is None) and tokens and (time.time() - last_time > PAUSE_SEC):
            # A pause is detected. Start a new correction job.
            sentence = " ".join(tokens)
            print(f"App: Pause detected. Sending tokens to correct: {tokens}")
            
            # Show "correcting" message
            OUTPUT.markdown(f"### 🗣 {sentence} (correcting...)")
            
            if gf:
                # Start the background thread
                correction_job["sentence"] = sentence
                correction_job["thread"] = Thread(target=correct_sentence_in_background,
                                                  args=(sentence, gf, correction_job))
                correction_job["thread"].start()
            else:
                # No Gramformer, just show the sentence
                OUTPUT.markdown(f"### 🗣 {sentence}")
            
            tokens.clear() # Clear tokens after starting job
        
        # 7. Draw overlay on frame
        if final_label:
            cv2.putText(frame, f"{final_label}", (30, 50),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 8. Display Frame
        FRAME.image(frame, channels="BGR")

    # Cleanup
    cap.release()
    hands.close()
    st.success("Session ended.")
