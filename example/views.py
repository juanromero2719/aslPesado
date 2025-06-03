# views.py  – versión PyTorch (.pth) SIN bordes negros
import os, tempfile, requests, base64, io
from pathlib import Path
from django.shortcuts import render
from PIL import Image
from django.http import JsonResponse

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np          # ← ya lo tenías instalado

from .forms import ImageUploadForm

# ──── Paths y constantes ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "models" / "labels_best.txt"

MODEL_URL   = (
    "https://www.dropbox.com/scl/fi/gq5o1ivzzunh8a8xexa2w/"
    "best_model.pth?rlkey=4jl4fol02qjwam7himq989ao6&st=4zd1fx50&dl=1"
)
MODEL_LOCAL = Path(tempfile.gettempdir()) / "best_model.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = [l.strip() for l in LABELS_PATH.read_text().splitlines()]

# ──── Función para recortar bordes negros ──────────────────────────────────────
def remove_black_borders(pil_img: Image.Image, thresh: int = 15) -> Image.Image:
    """
    Devuelve una nueva PIL sin columnas/filas cuya intensidad promedio
    sea < `thresh` (0-255).  Si toda la imagen es oscura se devuelve igual.
    """
    arr = np.asarray(pil_img)
    if arr.ndim == 2:                       # escala de grises
        gray = arr
    else:                                   # RGB
        gray = arr.mean(axis=2)

    mask = gray > thresh                   # True donde hay 'contenido'
    coords = np.argwhere(mask)

    if coords.size == 0:                   # imagen completamente oscura
        return pil_img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1        # +1 para incluir la fila/col final
    cropped = pil_img.crop((x0, y0, x1, y1))
    return cropped

# ──── Descarga y carga perezosa del modelo ─────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        if not MODEL_LOCAL.exists():
            print("⬇️  Descargando modelo…")
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            MODEL_LOCAL.write_bytes(r.content)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))

        state = torch.load(MODEL_LOCAL, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _model = model
    return _model

# ──── Preprocesamiento y normalización ─────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    # ① quitar bordes negros  ② convertir a tensor
    pil_img = remove_black_borders(pil_img)
    return _preprocess(pil_img).unsqueeze(0)          # shape (1,3,224,224)

# ──── Predicción top-k ─────────────────────────────────────────────────────────
@torch.no_grad()
def predict(pil_img: Image.Image, k: int = 5):
    logits = get_model()(preprocess(pil_img).to(DEVICE))[0]
    probs  = logits.softmax(dim=0).cpu().numpy()
    idx    = probs.argsort()[-k:][::-1]
    return [(labels[i], round(float(probs[i]) * 100, 2)) for i in idx]

# ──── Utilidad para mostrar preview en <img src="data:…"> ──────────────────────
def pil_to_base64(pil):
    pil = remove_black_borders(pil)                 # ← también para el preview
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ──── Vista principal ──────────────────────────────────────────────────────────
def upload_view(request):
    ctx = {"form": ImageUploadForm()}
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # 1️⃣ Obtención de la imagen
            if form.cleaned_data["rotated"]:
                b64 = form.cleaned_data["rotated"].split(",", 1)[1]
                pil_img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                ctx["img_data"] = form.cleaned_data["rotated"]
            elif form.cleaned_data["image"]:
                pil_img = Image.open(form.cleaned_data["image"]).convert("RGB")
                ctx["img_data"] = pil_to_base64(pil_img)   # sin bordes
            else:
                ctx["error"] = "Debes seleccionar una imagen."
                return render(request, "asl/upload.html", ctx)

            # 2️⃣ Inferencia
            try:
                ctx["preds"] = predict(pil_img)
            except Exception as e:
                ctx["error"] = "Error procesando la imagen."
                print("❌", e)

        ctx["form"] = form
    return render(request, "asl/upload.html", ctx)

def healthcheck(request):
    return JsonResponse({"status": "ok"})