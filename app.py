import faulthandler
import logging
import math
import os
import time

import cv2
from google import genai
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

faulthandler.enable()   # native crash par terminal mein wajah likh deta hai
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================== SETTINGS ===============================
GEMINI_MODEL = "gemini-3.5-flash-lite"   # model ka naam yahan badlein
CAMERA_INDEX = 0
W, H = 640, 480                          # har frame is size par resize hota hai

# ---- Hand / gesture ----
THUMB_RATIO = 1.1       # angutha "khula" tab maana jata hai (kam karein agar detect na ho)
PINCH_PX = 30           # thumb-index is se kareeb ho to pinch
DWELL_SEC = 0.8         # menu par itni der ruk kar select hota hai
RING_R = 40             # dwell circle ka shuru ka radius
COOLDOWN = 1.0          # save/undo ke darmiyan kam az kam waqfa (seconds)
TARGET_FPS = 20         # browser ko frames ka toofan na bheje (kam karein agar page atke)
SIDE_EVERY = 2          # canvas wali image har 2nd frame par update hoti hai

# ---- Paint ----
TOOLS = ["line", "rect", "draw", "circle", "erase"]
TOOL_W, TOOL_H = 50, 50
TOOLBAR_RECT = (150, 0, 150 + TOOL_W * len(TOOLS), TOOL_H)   # x0, y0, x1, y1
SAVE_RECT = (0, 0, 90, 40)
UNDO_RECT = (540, 0, 630, 40)
COLORS = {                                # BGR
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}
COLOR_NAMES = list(COLORS)
SWATCH_W, SWATCH_H = 60, 45
COLORBAR_RECT = (W - SWATCH_W, 60, W, 60 + SWATCH_H * len(COLOR_NAMES))
THICK = 4
ERASER_R = 30
MAX_HISTORY = 30
SAVE_FILE = "saved_canvas.png"

# ---- Math ----
MATH_COLOR = (255, 0, 255)
MATH_THICK = 10
MATH_PROMPT = ("Solve this handwritten math problem. Give the final answer first, "
               "then short step-by-step working.")

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================== HELPERS ===============================
def inside(pt, rect):
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def color_at(pt):
    """If there is a point on the color bar, return the swatch index; otherwise, return None."""
    if inside(pt, COLORBAR_RECT):
        idx = (pt[1] - COLORBAR_RECT[1]) // SWATCH_H
        if 0 <= idx < len(COLOR_NAMES):
            return int(idx)
    return None


def fingers_up(pts):
    """[thumb, index, middle, ring, pinky] -> 1 = upar, 0 = neeche."""
    thumb = int(dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * THUMB_RATIO)
    others = [int(pts[t][1] < pts[p][1]) for t, p in zip((8, 12, 16, 20), (6, 10, 14, 18))]
    return [thumb] + others


def is_pinch(pts):
    return dist(pts[4], pts[8]) < PINCH_PX


def detect_hand(hands, frame):
    """(21 landmark pixels, fingers) or None. Also draws landmarks on the frame."""
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
    pts = [(min(max(int(p.x * W), 0), W - 1), min(max(int(p.y * H), 0), H - 1)) for p in lm.landmark]
    return pts, fingers_up(pts)


def get_api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")


@st.cache_resource
def get_model(api_key):
    return genai.Client(api_key=api_key)


def load_toolbar():
    """If tools.png exists, use it at (250x50); otherwise, create a simple toolbar yourself."""
    size = (TOOL_W * len(TOOLS), TOOL_H)
    if os.path.exists("tools.png"):
        img = cv2.imread("tools.png")
        if img is not None:
            return cv2.resize(img, size)
    bar = np.full((size[1], size[0], 3), 235, np.uint8)
    for i, name in enumerate(TOOLS):
        x = i * TOOL_W
        cv2.rectangle(bar, (x, 0), (x + TOOL_W - 1, TOOL_H - 1), (120, 120, 120), 1)
        cv2.putText(bar, name[:4], (x + 5, 30), FONT, 0.45, (0, 0, 0), 1)
    return bar


# =============================== STATE ===============================
def new_paint_state():
    canvas = np.full((H, W, 3), 255, np.uint8)
    return {
        "canvas": canvas, "history": [canvas.copy()],
        "tool": "draw", "color": COLORS["black"],
        "stroke": False, "prev": None, "shape_start": None, "last_pt": None,
        "hover": None, "hover_t": 0.0, "hover_done": False,
        "last_action": 0.0, "msg": "", "msg_t": 0.0, "saves": 0, "png": None,
    }


def new_math_state():
    return {"canvas": np.zeros((H, W, 3), np.uint8), "prev": None,
            "last_gesture": None, "answer": "", "solve": False, "label": ""}


def init_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "paint"
        st.session_state.paint = new_paint_state()
        st.session_state.math = new_math_state()


def set_mode(mode):
    """When switching modes, the state of both modes is cleared."""
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.paint = new_paint_state()
        st.session_state.math = new_math_state()


def request_mode(mode):
    """Mode button click: if the camera is running, the mode does NOT change; it only shows a popup."""
    if st.session_state.get("run_cam", True):
        st.session_state["blocked_mode"] = mode
    else:
        set_mode(mode)


# =============================== VIRTUAL PAINT ===============================
def push_history(s):
    s["history"].append(s["canvas"].copy())
    if len(s["history"]) > MAX_HISTORY:
        s["history"].pop(0)


def draw_shape(img, tool, a, b, color, thick):
    if tool == "line":
        cv2.line(img, a, b, color, abs(thick) if thick > 0 else THICK)
    elif tool == "rect":
        cv2.rectangle(img, a, b, color, thick)
    elif tool == "circle":
        cv2.circle(img, a, int(dist(a, b)), color, thick)


def finish(s, fill, commit=True):
    """Ends the current stroke/shape. If commit=False, the shape is canceled"""
    if s["stroke"]:
        s["stroke"], s["prev"] = False, None
        push_history(s)
    if s["shape_start"] is not None:
        if commit and s["last_pt"] is not None and s["last_pt"] != s["shape_start"]:
            thick = -1 if fill and s["tool"] in ("rect", "circle") else THICK
            draw_shape(s["canvas"], s["tool"], s["shape_start"], s["last_pt"], s["color"], thick)
            push_history(s)
        s["shape_start"] = None


def dwell(s, target):
    """Progress for staying on this target, ranging from 0..1. Returns None after selection."""
    if target != s["hover"]:
        s["hover"], s["hover_t"], s["hover_done"] = target, time.time(), False
    if s["hover_done"]:
        return None
    p = (time.time() - s["hover_t"]) / DWELL_SEC
    if p >= 1.0:
        s["hover_done"] = True
        return 1.0
    return p


def draw_step(frame, s, p, fill):
    tool, color, canvas = s["tool"], s["color"], s["canvas"]
    s["last_pt"] = p
    if tool in ("draw", "erase"):
        if not s["stroke"]:                      # pehla point: (0,0) se line nahi banegi
            s["stroke"], s["prev"] = True, p
        if tool == "draw":
            cv2.line(canvas, s["prev"], p, color, THICK)
        else:
            cv2.line(canvas, s["prev"], p, (255, 255, 255), ERASER_R * 2)
            cv2.circle(canvas, p, ERASER_R, (255, 255, 255), -1)
            cv2.circle(frame, p, ERASER_R, (0, 0, 0), 2)
        s["prev"] = p
    else:                                        # line / rect / circle: preview frame par
        if s["shape_start"] is None:
            s["shape_start"] = p
        draw_shape(frame, tool, s["shape_start"], p, color, THICK)


def save_canvas(s):
    cv2.imwrite(SAVE_FILE, s["canvas"])
    s["png"] = cv2.imencode(".png", s["canvas"])[1].tobytes()
    s["saves"] += 1
    s["msg"], s["msg_t"] = "Saved!", time.time()
    logging.info("Canvas saved as %s", SAVE_FILE)


def undo(s):
    if len(s["history"]) > 1:
        s["history"].pop()
        s["canvas"] = s["history"][-1].copy()
        s["msg"] = "Undo!"
    else:
        s["msg"] = "Nothing to undo"
    s["msg_t"] = time.time()


def paint_step(frame, pts, fingers, s, fill):
    now = time.time()
    pointer = pts[8]
    drawing_pose = bool(fingers[1] and fingers[2])      # index + middle upar = likhna
    in_tools = inside(pointer, TOOLBAR_RECT)
    color_idx = color_at(pointer)
    in_save, in_undo = inside(pointer, SAVE_RECT), inside(pointer, UNDO_RECT)
    in_ui = in_tools or in_save or in_undo or color_idx is not None

    # --- tool / color select (dwell) ---
    target = None
    if in_tools:
        i = min((pointer[0] - TOOLBAR_RECT[0]) // TOOL_W, len(TOOLS) - 1)
        target = ("tool", TOOLS[i])
    elif color_idx is not None:
        target = ("color", COLOR_NAMES[color_idx])
    if target:
        prog = dwell(s, target)
        if prog is not None:
            cv2.circle(frame, pointer, max(4, int(RING_R * (1 - prog))), (0, 255, 255), 2)
            if prog >= 1.0:
                finish(s, fill, commit=False)
                if target[0] == "tool":
                    s["tool"] = target[1]
                else:
                    s["color"] = COLORS[target[1]]
    else:
        s["hover"], s["hover_done"] = None, False

    # --- save / undo (pinch + cooldown) ---
    if (in_save or in_undo) and is_pinch(pts) and now - s["last_action"] > COOLDOWN:
        s["last_action"] = now
        save_canvas(s) if in_save else undo(s)

    # --- drawing  ---
    if drawing_pose and not in_ui:
        draw_step(frame, s, pointer, fill)
    else:
        finish(s, fill, commit=True)


def draw_paint_ui(out, s, fill, toolbar):
    x0, y0, x1, y1 = TOOLBAR_RECT
    out[y0:y1, x0:x1] = cv2.addWeighted(toolbar, 0.7, out[y0:y1, x0:x1], 0.3, 0)
    i = TOOLS.index(s["tool"])
    cv2.rectangle(out, (x0 + i * TOOL_W, 0), (x0 + (i + 1) * TOOL_W, TOOL_H), (0, 0, 255), 2)
    cv2.putText(out, s["tool"], (x1 + 15, 30), FONT, 0.8, (0, 0, 255), 2)

    for rect, label, bg, fg in ((SAVE_RECT, "SAVE", (0, 255, 0), (0, 0, 255)),
                                (UNDO_RECT, "UNDO", (255, 0, 0), (255, 255, 255))):
        cv2.rectangle(out, rect[:2], rect[2:], bg, -1)
        cv2.putText(out, label, (rect[0] + 10, rect[1] + 28), FONT, 0.8, fg, 2)

    for idx, name in enumerate(COLOR_NAMES):
        sy = COLORBAR_RECT[1] + idx * SWATCH_H
        cv2.rectangle(out, (COLORBAR_RECT[0], sy), (W, sy + SWATCH_H), COLORS[name], -1)
        border = (255, 255, 255) if COLORS[name] == s["color"] else (100, 100, 100)
        cv2.rectangle(out, (COLORBAR_RECT[0], sy), (W, sy + SWATCH_H), border, 2)

    cv2.putText(out, "FILL: ON" if fill else "FILL: OFF", (10, H - 10), FONT, 0.7, (0, 128, 255), 2)
    if time.time() - s["msg_t"] < 1.5 and s["msg"]:
        cv2.putText(out, s["msg"], (100, 400), FONT, 1, (0, 0, 255), 2)


def run_paint(frame, hand, s, fill, toolbar):
    """Processes one frame -> (camera view, canvas view)."""
    if hand is None:                                   
        finish(s, fill, commit=False)
        s["hover"], s["hover_done"] = None, False
    else:
        paint_step(frame, hand[0], hand[1], s, fill)
    out = cv2.addWeighted(frame, 0.5, s["canvas"], 0.5, 0)
    draw_paint_ui(out, s, fill, toolbar)
    side = s["canvas"].copy()
    if hand:
        cv2.circle(side, hand[0][8], 6, (0, 0, 255), -1)
    return out, side


# =============================== MATH GESTURES ===============================
def run_math(frame, hand, m):
    """rocesses one frame -> (camera view, canvas view). Solve request is in m['solve']"""
    gesture = tuple(hand[1]) if hand else None
    label = ""
    if gesture == (0, 1, 0, 0, 0):                     
        p = hand[0][8]
        if m["prev"] is None:
            m["prev"] = p
        cv2.line(m["canvas"], m["prev"], p, MATH_COLOR, MATH_THICK)
        m["prev"] = p
        label = "Drawing"
    else:
        m["prev"] = None
    if gesture == (1, 0, 0, 0, 0):                     
        m["canvas"][:] = 0
        label = "Cleared"
    if gesture == (1, 1, 1, 1, 0):                     
        label = "Solve"
        if m["last_gesture"] != gesture:
            m["solve"] = True
    m["last_gesture"] = gesture                        
    out = cv2.addWeighted(frame, 0.7, m["canvas"], 0.3, 0)
    if label:
        cv2.putText(out, label, (10, 30), FONT, 0.9, (0, 255, 255), 2)
    return out, m["canvas"].copy()


def solve_with_ai(model, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) == 0:
        return "Canvas khali hai. Pehle ungli se sawal likhein."
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(255 - mask)                   # 
    try:
        return model.models.generate_content(model=GEMINI_MODEL, contents=[MATH_PROMPT, img]).text
    except Exception as e:
        logging.exception("Gemini error")
        return f"AI error: {e}"


def get_camera():
    """The camera opens only once and remains open even when switching modes.."""
    cap = st.session_state.get("cap")
    if cap is None or not cap.isOpened():
        flags = cv2.CAP_DSHOW if os.name == "nt" else 0
        cap = cv2.VideoCapture(CAMERA_INDEX, flags)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        st.session_state["cap"] = cap
    return cap


def release_camera():
    cap = st.session_state.pop("cap", None)
    if cap is not None:
        cap.release()


def get_hands():
    """The hand tracker is also created only once (repeatedly creating and closing it may cause crashes)."""
    if "hands" not in st.session_state:
        st.session_state["hands"] = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    return st.session_state["hands"]


# =============================== PAGE ===============================
MODE_LABELS = {"paint": "Virtual Paint", "math": "Math Gestures"}

if hasattr(st, "dialog"):
    @st.dialog("fiza First, turn off the camera.")
    def camera_popup(label):
        st.write(f"To switch modes, first untick **Camera Run** "
                 f"Then click the **{label}** button.")
else:                                                   # purani Streamlit mein dialog nahi hota
    def camera_popup(label):
        st.warning(f"Fiza first, untick 'Camera Run', then click the {label} button.")


def main():
    st.set_page_config(layout="wide", page_title="AI-GESTURE STUDIO")
    init_state()
    mode = st.session_state.mode
    st.title("AI-BASED HAND GESTURE CONTROL STUDIO")

    b1, b2 = st.columns(2)
    b1.button("Virtual Paint", type="primary" if mode == "paint" else "secondary",
              on_click=request_mode, args=("paint",), width="stretch")
    b2.button("Math Gestures", type="primary" if mode == "math" else "secondary",
              on_click=request_mode, args=("math",), width="stretch")

    blocked = st.session_state.pop("blocked_mode", None)
    if blocked:
        camera_popup(MODE_LABELS[blocked])

    run = st.checkbox("Camera Run", value=True, key="run_cam")
    fill = False
    model = None
    if mode == "paint":
        fill = st.checkbox("Fill shapes (rect / circle)", value=False)
        # st.caption(" Index + middle fingers up = draw   |  index finger only = pointer, hold over the menu for 0.8 sec "
        #            "to select a tool/color | thumb-index pinch = SAVE/UNDO")
    else:
        # st.caption("Index finger only = draw | thumb only = clear | thumb + index + middle + ring"
        #            "(pinky down) = solve")
        if os.path.exists("MathGestures.png"):
            with st.expander("Gestures guide"):
                st.image("MathGestures.png")
        key = get_api_key()
        if key:
            model = get_model(key)
        else:
            st.error(" Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml"
                     "or set it as an environment variable.")

    col_cam, col_side = st.columns([3, 2])
    frame_slot = col_cam.empty()
    side_slot = col_side.empty()
    text_slot = col_side.empty()

    if not run:
        release_camera()
        st.info(f"Camera is off. Turn on 'Camera Run' to start {MODE_LABELS[mode]} mode.")
        return

    cap = get_camera()
    if not cap.isOpened():
        release_camera()
        st.error("Camera did not open. Check that no other app is using the camera.")
        return
    hands = get_hands()

    s, m = st.session_state.paint, st.session_state.math
    toolbar = load_toolbar()
    shown_saves, shown_answer, fails, n = -1, None, 0, 0

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                release_camera()
                st.error("Unable to get a frame from the camera.")
                break
            time.sleep(0.03)
            continue
        fails = 0
        frame = cv2.flip(cv2.resize(frame, (W, H)), 1)

        try:
            hand = detect_hand(hands, frame)
            if mode == "paint":
                out, side = run_paint(frame, hand, s, fill, toolbar)
            else:
                out, side = run_math(frame, hand, m)
        except Exception:
            logging.exception("Frame could not be processed.")
            out, side = frame, None

        frame_slot.image(out, channels="BGR", output_format="JPEG", width="stretch")
        n += 1
        if side is not None and n % SIDE_EVERY == 0:
            side_slot.image(side, channels="BGR", output_format="JPEG", width="stretch")

        if mode == "paint":
            if s["saves"] > 0 and s["saves"] != shown_saves:
                shown_saves = s["saves"]
                text_slot.download_button("Download canvas (PNG)", s["png"], "canvas.png",
                                          "image/png", key=f"dl_{shown_saves}")
        else:
            if m["solve"]:
                m["solve"] = False
                if model is None:
                    m["answer"] = "API key is not set fiza, so the problem could not be solved.."
                else:
                    text_slot.info("Solving...")
                    m["answer"] = solve_with_ai(model, m["canvas"])
            if m["answer"] != shown_answer:
                shown_answer = m["answer"]
                text_slot.markdown(m["answer"] or "Nothing to answer yet. Write a question and raise 4 fingers fiza.")
        time.sleep(max(0.0, 1.0 / TARGET_FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()









 