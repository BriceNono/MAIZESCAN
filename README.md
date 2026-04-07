# MaizeScan — Maize Leaf Disease Detection

**Graduation Project** | CNN Transfer Learning with VGG16 | Deployed on Render

Detects: **Common Rust** · **Gray Leaf Spot** · **Northern Leaf Blight** · **Healthy**

---

## Project structure

```
maizescan/
├── app.py                  ← Flask backend + inference logic
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
├── Procfile                ← Gunicorn start command
├── runtime.txt             ← Python version pin
├── .gitignore
├── model/
│   └── vgg16_maize_best.h5 ← Your trained model (add this!)
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── css/style.css       ← Styles
    └── js/app.js           ← Frontend logic
```

---

## Step 1 — Add your trained model

After training in Google Colab, download the model file and place it here:

```
maizescan/model/vgg16_maize_best.h5
```

If the model file is missing, the app runs in **demo mode** — it still works and shows realistic simulated predictions.

> **Note:** The `.gitignore` excludes `.h5` files from git (they are large). For Render deployment, use one of the options in Step 4.

---

## Step 2 — Run locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Open http://localhost:5000 in your browser.

---

## Step 3 — Test the API

```bash
# Health check
curl http://localhost:5000/health

# Predict from a leaf image
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/maize_leaf.jpg"
```

**Response example:**
```json
{
  "success": true,
  "prediction": "Common Rust",
  "confidence": 0.9131,
  "confidence_pct": "91.3%",
  "scientific": "Puccinia sorghi",
  "severity": "Moderate",
  "spread": "Wind-borne spores — spreads rapidly",
  "description": "Small, circular to oval, powdery pustules...",
  "actions": ["Apply triazole fungicide...", "..."],
  "prevention": "Use rust-resistant varieties...",
  "all_probs": {
    "Northern Leaf Blight": 0.0041,
    "Common Rust": 0.9131,
    "Gray Leaf Spot": 0.0033,
    "Healthy": 0.0013
  },
  "image_preview": "data:image/jpeg;base64,..."
}
```

---

## Step 4 — Deploy to Render

### Option A — Model included in repo (small model only)

If your `.h5` file is under ~100 MB (unlikely for VGG16), you can include it:

1. Remove `*.h5` and `model/` from `.gitignore`
2. Commit and push with the model file
3. Connect repo to Render → auto-deploys

### Option B — Download model at build time (recommended)

Upload your `.h5` to Google Drive or any public URL, then add a `build.sh`:

```bash
#!/bin/bash
pip install -r requirements.txt
mkdir -p model
# Replace with your public download URL
wget -O model/vgg16_maize_best.h5 \
  "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
```

Update `render.yaml`:
```yaml
buildCommand: bash build.sh
```

To get your Google Drive file ID:
1. Right-click the `.h5` file in Drive → Get link → Change to "Anyone with the link"
2. Copy the ID from the URL: `https://drive.google.com/file/d/THIS_IS_THE_ID/view`

### Option C — Use Render Persistent Disk (paid plans)

Upload your model to the disk, reference it via `MODEL_PATH` env var.

### Render deployment steps

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial MaizeScan deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/maizescan.git
git push -u origin main
```

2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Create Web Service**
5. Wait for build (~5–10 min first time due to TensorFlow install)
6. Your app is live at `https://maizescan.onrender.com`

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model/vgg16_maize_best.h5` | Path to trained Keras model |
| `PORT` | `5000` | Server port (Render sets this automatically) |
| `FLASK_ENV` | `production` | Flask environment |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Upload leaf image → get diagnosis |
| `GET` | `/classes` | List all detectable disease classes |

---

## Troubleshooting

**"Model not found, running in demo mode"**
→ Place your `.h5` file at `model/vgg16_maize_best.h5` or set the `MODEL_PATH` env variable.

**Render build fails with memory error**
→ TensorFlow is large. On free tier, try `tensorflow-cpu` instead of `tensorflow` in requirements.txt.

**Slow first prediction**
→ Normal. The model loads on first request (~10–20s). Subsequent predictions are fast.

**"Module not found: flask_cors"**
→ Run `pip install flask-cors` or check requirements.txt is complete.

---

## Tech stack

| Layer | Technology |
|---|---|
| Model | VGG16 + custom head, trained with Keras/TF |
| Backend | Flask 3.0 + Gunicorn |
| Frontend | Vanilla HTML/CSS/JS — no framework required |
| Deployment | Render (free tier compatible) |
| Fonts | Playfair Display + DM Sans + DM Mono |

---

*Graduation project — Data Science / Computer Science · 2025–2026*
# MAIZESCAN
