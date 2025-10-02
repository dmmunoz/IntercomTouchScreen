# /opt/intercom/main.py — v1.2.7
# - OUT8 = "TODOS": TALK ALL envía el mic (IN1) a todas las salidas (OUT1..OUT7) con talkover
# - LISTEN por canal (según tu petición): al pulsar LISTEN en OUTi, se envía la **entrada** configurada
#   (listen_in[i]) a la **salida física** configurada (listen_out_phys[i]) con suavizado (no monitor post‑mix).
# - TALK y LISTEN con suavizado; vúmetros post‑mix lógico (sin incluir la inyección LISTEN).
#
# Ejecuta: /opt/intercom/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080

import os, json, asyncio, threading
from typing import List, Set, Tuple
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------- Audio ----------------
SR = 48000
BLOCK = 1024
LATENCY_TAG = "high"
NUM_INPUTS = 8
NUM_OUTPUTS = 8            # OUT1..OUT7 normales, OUT8=Todos
ALL_IDX = NUM_OUTPUTS-1
MIC_IDX = 0                # IN1 mic
BED_IDX = 1                # IN2 bed
DEV_HINT = "UMC"

CROSSFADE_MS = 35.0
HEADROOM = 0.85
SOFTCLIP_DRIVE = 1.2

# ---------------- Estado ----------------
_state_lock = threading.Lock()
_talking: List[bool] = [False] * NUM_OUTPUTS
_listening: List[bool] = [False] * NUM_OUTPUTS

_labels: List[str] = [f"Canal {i+1}" for i in range(NUM_OUTPUTS)]
_labels[ALL_IDX] = "TODOS"

# Matriz OUTxIN (por defecto: Bed a todo)
_matrix = np.zeros((NUM_OUTPUTS, NUM_INPUTS), dtype=np.float32)
_matrix[:, BED_IDX] = 1.0

# Ruteo lógico->físico (OUTi -> phys channel); OUT8 no se rutea (control)
OUT_MAP: List[int] = list(range(NUM_OUTPUTS))

# LISTEN por canal: entrada y salida física
LISTEN_IN: List[int] = [min(i, NUM_INPUTS-1) for i in range(NUM_OUTPUTS)]  # por defecto IN(i+1)
LISTEN_OUT_PHYS: List[int] = [0 for _ in range(NUM_OUTPUTS)]               # por defecto CH1
LISTEN_ENA: List[bool] = [True]*(NUM_OUTPUTS-1) + [False]                  # OUT8 sin listen

# Suavizados por salida
_talk_smooth = np.zeros(NUM_OUTPUTS, dtype=np.float32)
_listen_smooth = np.zeros(NUM_OUTPUTS, dtype=np.float32)

# Vúmetros (post-mix lógico)
_last_levels = np.zeros(NUM_OUTPUTS, dtype=np.float32)

# Conexiones WS
_clients: Set[WebSocket] = set()

# Paths
BASE_DIR = os.path.dirname(__file__)
STATE_FILE = os.path.join(BASE_DIR, "state.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
PHYS_IN_CH = None
PHYS_OUT_CH = None

# ---------------- Persistencia ----------------
def load_state():
    global _labels, _matrix, OUT_MAP, LISTEN_IN, LISTEN_OUT_PHYS, LISTEN_ENA
    try:
        with open(STATE_FILE, "r") as f: data = json.load(f)
        labs = data.get("channels")
        if isinstance(labs, list):
            if labs and isinstance(labs[0], dict):
                tmp = [None]*NUM_OUTPUTS
                for it in labs:
                    i = (it.get("id") or 0) - 1
                    if 0 <= i < NUM_OUTPUTS: tmp[i] = it.get("label")
                _labels[:] = [tmp[i] or _labels[i] for i in range(NUM_OUTPUTS)]
            else:
                L = [str(x) for x in labs]
                for i in range(min(NUM_OUTPUTS, len(L))): _labels[i] = L[i]
        _labels[ALL_IDX] = _labels[ALL_IDX] or "TODOS"
        mat = data.get("matrix")
        if isinstance(mat, list) and len(mat) >= NUM_OUTPUTS and len(mat[0]) >= NUM_INPUTS:
            _matrix[:, :] = np.array(mat, dtype=np.float32)[:NUM_OUTPUTS, :NUM_INPUTS]
        om = data.get("out_map")
        if isinstance(om, list) and len(om) >= NUM_OUTPUTS:
            OUT_MAP[:] = [int(x) for x in om[:NUM_OUTPUTS]]
        li = data.get("listen_in")
        if isinstance(li, list) and len(li) >= NUM_OUTPUTS:
            LISTEN_IN[:] = [max(0, min(int(x), NUM_INPUTS-1)) for x in li[:NUM_OUTPUTS]]
        lo = data.get("listen_out_phys")
        if isinstance(lo, list) and len(lo) >= NUM_OUTPUTS:
            LISTEN_OUT_PHYS[:] = [max(0,int(x)) for x in lo[:NUM_OUTPUTS]]
        le = data.get("listen_enabled")
        if isinstance(le, list) and len(le) >= NUM_OUTPUTS:
            LISTEN_ENA[:] = [bool(x) for x in le[:NUM_OUTPUTS]]
            LISTEN_ENA[ALL_IDX] = False
    except Exception:
        pass

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "channels": [{"id":i+1,"label":_labels[i]} for i in range(NUM_OUTPUTS)],
                "matrix": _matrix.tolist(),
                "out_map": OUT_MAP,
                "listen_in": LISTEN_IN,
                "listen_out_phys": LISTEN_OUT_PHYS,
                "listen_enabled": LISTEN_ENA
            }, f)
        return True
    except Exception:
        return False

# ---------------- Dispositivo ----------------
def _pick_device_indices(hint: str) -> Tuple[int,int]:
    devs = sd.query_devices(); in_idx = out_idx = None
    for i,d in enumerate(devs):
        name=(d.get("name") or "").lower()
        if hint.lower() in name:
            if d.get("max_input_channels",0)>=2 and in_idx is None: in_idx=i
            if d.get("max_output_channels",0)>=2 and out_idx is None: out_idx=i
    if in_idx is None:
        in_idx = max(range(len(devs)), key=lambda i:int(devs[i].get("max_input_channels",0)))
    if out_idx is None:
        out_idx = max(range(len(devs)), key=lambda i:int(devs[i].get("max_output_channels",0)))
    return in_idx, out_idx

def _probe_channels(devpair: Tuple[int,int]) -> Tuple[int,int]:
    global PHYS_IN_CH, PHYS_OUT_CH
    info_in = sd.query_devices(devpair[0]); info_out = sd.query_devices(devpair[1])
    max_in = int(info_in.get("max_input_channels",0)); max_out = int(info_out.get("max_output_channels",0))
    in_try = [NUM_INPUTS, 10, 12, 16]; in_try = [x for x in in_try if x<=max_in] or [max_in]
    out_try= [12, 10, NUM_OUTPUTS, 16, 18, 20]; out_try=[x for x in out_try if x<=max_out] or [max_out]
    for nin in in_try:
        for nout in out_try:
            try:
                st = sd.Stream(samplerate=SR, blocksize=BLOCK, dtype="float32",
                               device=devpair, channels=(nin, nout), latency=LATENCY_TAG,
                               callback=lambda *a, **k: None)
                st.close(); PHYS_IN_CH, PHYS_OUT_CH = nin, nout; return nin, nout
            except Exception: continue
    PHYS_IN_CH, PHYS_OUT_CH = min(max_in, NUM_INPUTS), min(max_out, NUM_OUTPUTS)
    return PHYS_IN_CH, PHYS_OUT_CH

# ---------------- DSP ----------------
def softclip(x: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 1.0: return x
    y = np.tanh(x * drive)
    return y / np.tanh(drive)

def audio_callback(indata, outdata, frames, time_info, status):
    X_src = indata.astype(np.float32)
    if X_src.shape[1] >= NUM_INPUTS:
        X = X_src[:, :NUM_INPUTS]
    else:
        X = np.zeros((frames, NUM_INPUTS), dtype=np.float32)
        X[:, :X_src.shape[1]] = X_src

    with _state_lock:
        M = _matrix.copy()
        route = list(OUT_MAP)
        target_talk = np.array(_talking, dtype=np.float32)
        target_listen = np.array(_listening, dtype=np.float32)
        li = list(LISTEN_IN)
        lo = list(LISTEN_OUT_PHYS)

        alpha = 1.0 - float(np.exp(- (frames / SR) / (CROSSFADE_MS/1000.0)))
        alpha = float(min(1.0, max(0.0, alpha)))
        global _talk_smooth, _listen_smooth
        _talk_smooth   += alpha * (target_talk   - _talk_smooth)
        _listen_smooth += alpha * (target_listen - _listen_smooth)
        g  = _talk_smooth.copy()      # 0..1 (TALK)
        gl = _listen_smooth.copy()    # 0..1 (LISTEN)

    # TALK ALL
    g_all = float(g[ALL_IDX])
    g_eff = g.copy(); g_eff[ALL_IDX] = 0.0
    for i in range(NUM_OUTPUTS-1):
        g_eff[i] = max(g_eff[i], g_all)

    # Mezcla lógica base sin mic/bed
    M_other = M.copy()
    M_other[:, MIC_IDX] = 0.0; M_other[:, BED_IDX] = 0.0
    base = X @ M_other.T

    bed_gain = M[:, BED_IDX]
    mic_term = X[:, MIC_IDX][:, None] * g_eff[None, :]
    bed_term = X[:, BED_IDX][:, None] * (bed_gain[None,:] * (1.0 - g_eff[None,:]))

    out_logic = base + mic_term + bed_term
    out_logic[:, ALL_IDX] = 0.0
    out_logic *= HEADROOM
    out_logic = softclip(out_logic, SOFTCLIP_DRIVE)
    np.clip(out_logic, -1.0, 1.0, out=out_logic)

    # Volcado lógico->físico (ignora OUT ALL)
    outdata.fill(0.0)
    phys_out = outdata.shape[1]
    for i in range(NUM_OUTPUTS):
        if i == ALL_IDX: continue
        p = route[i]
        if 0 <= p < phys_out:
            outdata[:, p] += out_logic[:, i]

    # Inyección LISTEN: enviar la **entrada** configurada de cada OUTi a su salida física
    for i in range(NUM_OUTPUTS):
        if not LISTEN_ENA[i]: continue
        if gl[i] <= 0.0: continue
        p = lo[i]
        if 0 <= p < phys_out:
            src = max(0, min(li[i], X.shape[1]-1))
            outdata[:, p] += (X[:, src] * gl[i] * 0.9)  # 0.9 = headroom listen

    np.clip(outdata, -1.0, 1.0, out=outdata)

    # Vúmetros de salidas lógicas
    rms = np.sqrt(np.mean(out_logic * out_logic, axis=0, dtype=np.float32))
    with _state_lock:
        global _last_levels
        _last_levels = 0.6*_last_levels + 0.4*np.clip(rms, 0.0, 1.0)

# ---------------- API ----------------
app = FastAPI(title="Intercom v1.2.7 (LISTEN por entrada por canal)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/config")
def api_config():
    with _state_lock:
        t = list(_talking); l = list(_listening); po = PHYS_OUT_CH
    return {"sr": SR, "block": BLOCK, "num_inputs": NUM_INPUTS, "num_outputs": NUM_OUTPUTS,
            "mic_index": MIC_IDX, "bed_index": BED_IDX, "talking": t, "listening": l,
            "phys_out": po, "all_index": ALL_IDX, "listen_supported": True}

@app.get("/api/state")
def api_state_get():
    with _state_lock:
        labs = list(_labels); M = _matrix.tolist()
    return {"channels":[{"id":i+1,"label":labs[i]} for i in range(NUM_OUTPUTS)],
            "matrix":M, "mic_index":MIC_IDX, "bed_index":BED_IDX,
            "num_inputs":NUM_INPUTS, "num_outputs":NUM_OUTPUTS, "all_index": ALL_IDX}

@app.post("/api/state")
def api_state_set(payload: dict):
    chs = payload.get("channels")
    if isinstance(chs, list):
        with _state_lock:
            for it in chs:
                try:
                    i = int(it.get("id",0)) - 1
                    if 0 <= i < NUM_OUTPUTS:
                        _labels[i] = str(it.get("label", _labels[i]))
                except Exception: pass
        if not _labels[ALL_IDX]: _labels[ALL_IDX] = "TODOS"
        save_state()
    return {"ok":True}

@app.get("/api/matrix")
def api_matrix_get():
    with _state_lock: M = _matrix.tolist()
    return {"matrix":M, "num_inputs":NUM_INPUTS, "num_outputs":NUM_OUTPUTS,
            "mic_index":MIC_IDX, "bed_index":BED_IDX, "all_index": ALL_IDX}

@app.post("/api/matrix")
def api_matrix_set(payload: dict):
    mat = payload.get("matrix")
    if not (isinstance(mat, list) and len(mat) == NUM_OUTPUTS and len(mat[0]) == NUM_INPUTS):
        return JSONResponse({"ok":False,"error":f"matrix shape must be {NUM_OUTPUTS}x{NUM_INPUTS}"}, status_code=400)
    with _state_lock:
        _matrix[:, :] = np.array(mat, dtype=np.float32)
    save_state(); return {"ok":True}

@app.get("/api/routing")
def api_routing_get():
    with _state_lock: om = list(OUT_MAP); po = PHYS_OUT_CH
    return {"map":om, "phys_out":po, "all_index": ALL_IDX}

@app.post("/api/routing")
def api_routing_set(payload: dict):
    m = payload.get("map")
    if not (isinstance(m, list) and len(m) == NUM_OUTPUTS):
        return JSONResponse({"ok":False,"error":f"map length must be {NUM_OUTPUTS}"}, status_code=400)
    try: mm = [int(x) for x in m]
    except Exception: return JSONResponse({"ok":False,"error":"map must be integers"}, status_code=400)
    with _state_lock:
        for i in range(NUM_OUTPUTS): mm[i] = max(0,int(mm[i]))
        OUT_MAP[:] = mm
    save_state(); return {"ok":True,"map":OUT_MAP,"all_index":ALL_IDX}

@app.get("/api/listen_config")
def api_listen_get():
    with _state_lock:
        li = list(LISTEN_IN); lo = list(LISTEN_OUT_PHYS); le = list(LISTEN_ENA); lstate = list(_listening)
        po = PHYS_OUT_CH
    return {"listen_in": li, "listen_out_phys": lo, "listen_enabled": le, "listening": lstate, "phys_out": po, "all_index": ALL_IDX}

@app.post("/api/listen_config")
def api_listen_set(payload: dict):
    li = payload.get("listen_in"); lo = payload.get("listen_out_phys"); le = payload.get("listen_enabled")
    with _state_lock:
        if isinstance(li, list) and len(li) == NUM_OUTPUTS:
            for i in range(NUM_OUTPUTS):
                LISTEN_IN[i] = max(0, min(int(li[i]), NUM_INPUTS-1))
        if isinstance(lo, list) and len(lo) == NUM_OUTPUTS:
            for i in range(NUM_OUTPUTS):
                LISTEN_OUT_PHYS[i] = max(0, int(lo[i]))
        if isinstance(le, list) and len(le) == NUM_OUTPUTS:
            for i in range(NUM_OUTPUTS):
                LISTEN_ENA[i] = bool(le[i])
            LISTEN_ENA[ALL_IDX] = False
    save_state(); return {"ok": True}

@app.post("/api/listen")
async def api_listen_toggle(payload: dict):
    idv = payload.get("id"); state = bool(payload.get("state"))
    if idv is None: return JSONResponse({"ok":False,"error":"missing id"}, status_code=400)
    idx = int(idv); idx = idx-1 if 1 <= idx <= NUM_OUTPUTS else idx
    if not (0 <= idx < NUM_OUTPUTS): return JSONResponse({"ok":False,"error":"bad channel"}, status_code=400)
    with _state_lock:
        if not LISTEN_ENA[idx]:
            _listening[idx] = False
        else:
            _listening[idx] = state    # NO exclusivo
    await broadcast({"type":"listen_state","listening":list(_listening)})
    return {"ok":True}

@app.get("/api/meters")
def api_meters():
    with _state_lock: lv = [float(x) for x in _last_levels]
    return {"levels": lv}

@app.post("/api/talk")
async def api_talk(payload: dict):
    idv = payload.get("id"); state = bool(payload.get("state"))
    if idv is None: return JSONResponse({"ok":False,"error":"missing id"}, status_code=400)
    idx = int(idv); idx = idx-1 if 1 <= idx <= NUM_OUTPUTS else idx
    if not (0 <= idx < NUM_OUTPUTS): return JSONResponse({"ok":False,"error":"bad channel"}, status_code=400)
    with _state_lock: _talking[idx] = state
    await broadcast({"type":"state","channels":list(_talking)})
    return {"ok":True}

# ---------------- WS ----------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept(); _clients.add(websocket)
    try:
        await websocket.send_json({"type":"hello","role":"server","channels":NUM_OUTPUTS,"all_index":ALL_IDX})
        with _state_lock:
            t = list(_talking); l = list(_listening); lv = [float(x) for x in _last_levels]
        await websocket.send_json({"type":"state","channels":t})
        await websocket.send_json({"type":"listen_state","listening":l})
        await websocket.send_json({"type":"meters","levels":lv})
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if data.get("type") == "talk":
                    idv = int(data.get("id")); st = bool(data.get("state"))
                    idv = idv-1 if 1 <= idv <= NUM_OUTPUTS else idv
                    if 0 <= idv < NUM_OUTPUTS:
                        with _state_lock: _talking[idv] = st
                        await broadcast({"type":"state","channels":list(_talking)})
                elif data.get("type") == "listen":
                    idv = int(data.get("id")); st = bool(data.get("state"))
                    idv = idv-1 if 1 <= idv <= NUM_OUTPUTS else idv
                    if 0 <= idv < NUM_OUTPUTS and LISTEN_ENA[idv]:
                        with _state_lock: _listening[idv] = st
                        await broadcast({"type":"listen_state","listening":list(_listening)})
            except Exception: pass
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)

async def broadcast(obj: dict):
    dead = []
    for ws in list(_clients):
        try: await ws.send_json(obj)
        except Exception: dead.append(ws)
    for ws in dead: _clients.discard(ws)

async def _meters_task():
    while True:
        await asyncio.sleep(0.1)
        try:
            with _state_lock:
                lv = [float(x) for x in _last_levels]
            if lv: await broadcast({"type":"meters","levels":lv})
        except Exception: pass

# ---------------- Arranque audio ----------------
_audio_stream = None
_stop_flag = False

def _prefer_alsa():
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "alsa" in (api.get("name") or "").lower():
                sd.default.hostapi = i; break
    except Exception: pass

async def audio_start_loop():
    global _audio_stream, PHYS_IN_CH, PHYS_OUT_CH
    while not _stop_flag:
        try:
            _prefer_alsa()
            in_idx, out_idx = _pick_device_indices(DEV_HINT)
            sd.default.samplerate = SR
            sd.default.blocksize = BLOCK
            sd.default.dtype = "float32"
            sd.default.device = (in_idx, out_idx)
            _probe_channels((in_idx, out_idx))

            stream = sd.Stream(samplerate=SR, blocksize=BLOCK, dtype="float32",
                               device=(in_idx, out_idx),
                               channels=(PHYS_IN_CH, PHYS_OUT_CH),
                               latency=LATENCY_TAG,
                               callback=audio_callback)
            stream.start(); _audio_stream = stream
            print(f"[audio] IN#{in_idx}/OUT#{out_idx} @ {SR} Hz, block {BLOCK}, latency={LATENCY_TAG} (phys {PHYS_IN_CH} in / {PHYS_OUT_CH} out)")
            return
        except Exception as e:
            print(f"[audio] Falló inicio: {e}. Reintentando en 2s..."); await asyncio.sleep(2)

# ---------------- FastAPI hooks ----------------
@app.on_event("startup")
async def on_start():
    load_state()
    asyncio.create_task(_meters_task())
    asyncio.create_task(audio_start_loop())

@app.on_event("shutdown")
async def on_stop():
    global _stop_flag
    _stop_flag = True
    if _audio_stream:
        try: _audio_stream.stop(); _audio_stream.close()
        except Exception: pass
