import asyncio
import csv
import json
import uuid
from typing import Dict, Any
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

from fastapi import FastAPI, HTTPException, Request, Form, Depends, Cookie, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
from contextlib import asynccontextmanager

GNN_MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GNN_MODEL
    try:
        if os.path.exists("aiops_gnn_model.pth"):
            GNN_MODEL = torch.load("aiops_gnn_model.pth", map_location='cpu', weights_only=False)
            print("✅ [MODEL] aiops_gnn_model.pth başarıyla yüklendi ve çıkarıma hazır.")
    except Exception as e:
        print(f"Failed to load aiops_gnn_model.pth: {e}")
        
    asyncio.create_task(simulate_data_stream())
    print("🚀 [SERVER] AIOps Platformu http://127.0.0.1:5001 adresinde yayında!")
    
    yield
    # Burası shutdown kısmı
    # print("Sunucu kapanıyor...")

app = FastAPI(title="NEXUS AI Anomaly Detection Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Set up templates
templates = Jinja2Templates(directory=str(BASE_DIR))

import hmac
import hashlib
import base64

SECRET_KEY = "NEXUS_AIOPS_SUPER_SECRET_KEY"

def create_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(SECRET_KEY.encode(), f"{header}.{payload_b64}".encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return f"{header}.{payload_b64}.{signature}"

def decode_jwt(token: str) -> dict:
    try:
        header, payload_b64, signature = token.split(".")
        expected_sig = base64.urlsafe_b64encode(hmac.new(SECRET_KEY.encode(), f"{header}.{payload_b64}".encode(), hashlib.sha256).digest()).decode().rstrip("=")
        if hmac.compare_digest(signature, expected_sig):
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            return json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    except Exception:
        pass
    return None

# Users and Sessions
USERS = {
    "demo@nexus.ai": {"password": "nexus123", "name": "Demo User"}
}
sessions: Dict[str, str] = {}  # session_id -> email

def get_current_user(request: Request):
    # JWT Auth Guard Dependency
    token = request.cookies.get("session_token")
    auth_header = request.headers.get("Authorization")
    
    # Check Bearer also if checking API
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        
    user = None
    if token:
        # Check if JWT
        payload = decode_jwt(token)
        if payload and "email" in payload:
            user = payload["email"]
        elif token in sessions:
            user = sessions[token]

    if not user:
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT, headers={"Location": "/login"})
        
    return user

class RequiresLoginException(Exception):
    pass

@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    # Route protected page requests to login page
    if exc.status_code == status.HTTP_307_TEMPORARY_REDIRECT and "Location" in exc.headers:
        return RedirectResponse(url="/login")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Serve HTML files directly via endpoints to enforce auth logic
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/register")
async def api_register(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
        
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email ve şifre zorunludur")
        
    if email in USERS:
        raise HTTPException(status_code=400, detail="Bu e-posta adresi zaten kullanımda")
        
    USERS[email] = {"password": "nexus123" if password is None else password, "name": name or "New User"}
    return {"success": True, "message": "Kayıt başarılı"}

@app.post("/api/login")
async def api_login(request: Request):
    try:
        data = await request.json()
    except Exception:
        # Fallback for Form data
        form = await request.form()
        data = {"email": form.get("email"), "password": form.get("password")}
        
    email = data.get("email")
    password = data.get("password")
    
    if email in USERS and password == USERS[email]["password"]:
        jwt_token = create_jwt({"email": email, "name": USERS[email]["name"]})
        
        response = JSONResponse(content={"success": True, "redirect": "/dashboard", "token": jwt_token})
        response.set_cookie(key="session_token", value=jwt_token, httponly=True)
        return response
    
    raise HTTPException(status_code=400, detail="Hatalı e-posta veya şifre")

@app.post("/api/logout")
async def api_logout(request: Request):
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_token")
    return response

# Protected Routes
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/network", response_class=HTMLResponse)
async def network(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("network.html", {"request": request})

@app.get("/iot", response_class=HTMLResponse)
async def iot(request: Request, user: str = Depends(get_current_user)):
    # Fallback in case actual file is ıot.html
    try:
        return templates.TemplateResponse("iot.html", {"request": request})
    except Exception:
         return templates.TemplateResponse("ıot.html", {"request": request})

@app.get("/finance", response_class=HTMLResponse)
async def finance(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("finance.html", {"request": request})

@app.get("/ik", response_class=HTMLResponse)
async def ik(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("ik.html", {"request": request})


# WEBSOCKET REAL-TIME ENGINE
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

def load_data():
    csv_path = BASE_DIR / "enterprise_telemetry_data.csv"
    if not csv_path.exists():
        return []
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

TELEMETRY_DATA = load_data()

async def simulate_data_stream():
    if not TELEMETRY_DATA:
        print("No telemetry data found!")
        return

    idx = 0
    while True:
        # Get next row wrapping around
        row = TELEMETRY_DATA[idx % len(TELEMETRY_DATA)]
        
        # GNN Model inference simulation using PyTorch model state if loaded
        # In a real scenario, we'd package 'row' into a node feature matrix and adjacency matrix for the GNN
        
        label = int(row.get("label", 0))
        
        # Calculate simulated anomaly score
        # Using model presence as a modifier for simulation
        base_score = 0.85 if label == 1 else 0.15
        
        # We perturb the score with pseudo-weights based on network conditions
        if GNN_MODEL is not None:
            # Simulated PyTorch GNN projection
            tensor_val = torch.tensor(float(row.get("hub_cpu", 0)) / 100).float()
            gnn_activation = torch.sigmoid(tensor_val).item() * 0.4
            anomaly_score = min(max(base_score + (gnn_activation - 0.2), 0.0), 1.0)
        else:
            anomaly_score = base_score

        # If 'label' == 1, generate anomaly alerts
        alerts = []
        if anomaly_score >= 0.7:  # Critical
            status = 'critical'
            alerts.append({"dept": "iot", "issue": "Critical Flow", "severity": "critical"})
            if float(row.get("it_server_resp_time", 0)) > 200:
                alerts.append({"dept": "network", "issue": "High Latency", "severity": "critical"})
        elif anomaly_score >= 0.4:  # Warning
            status = 'warning'
            alerts.append({"dept": "finance", "issue": "Volume Spikes", "severity": "warning"})
        else:
            status = 'stable'

        payload = {
            "type": "telemetry",
            "data": row,
            "anomaly_score": round(anomaly_score, 2),
            "status": status,
            "is_anomaly": anomaly_score >= 0.7,
            "alerts": alerts
        }
        
        try:
            await manager.broadcast(payload)
        except Exception:
            pass
            
        idx += 1
        await asyncio.sleep(1.0)  # "Streaming once a second" as requested


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, session_token: str = Cookie(None)):
    # Simple websocket auth via JWT or fallback
    is_authenticated = False
    if session_token:
        payload = decode_jwt(session_token)
        if payload and "email" in payload:
            is_authenticated = True
        elif session_token in sessions:
            is_authenticated = True
            
    if not is_authenticated:
        await websocket.close(code=1008)
        return
        
    await manager.connect(websocket)
    try:
        while True:
            # Keep alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    import socket
    
    PORT = 5001
    
    # Portun meşgul olup olmadığını kontrol et
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_port_in_use = s.connect_ex(('127.0.0.1', PORT)) == 0
        
    if is_port_in_use:
        print(f"❌ [HATA] {PORT} portu şu anda başka bir süreç tarafından meşgul ediliyor (Errno 10048).")
    else:
        try:
            uvicorn.run(app, host="127.0.0.1", port=PORT)
        except Exception as e:
            print(f"❌ [HATA] Sunucu başlatılamadı: {e}")
