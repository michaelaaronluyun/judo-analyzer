# DLSU Judo Analyzer — Local & Vercel Deployment Guide

Migrated from Google Colab notebook to a proper project structure.

---

## Project Structure

```
judo-analyzer/
├── api/                    ← Vercel Python serverless functions
│   ├── auth.py             → /api/auth?action=...
│   ├── analyze.py          → /api/analyze
│   ├── heatmap.py          → /api/heatmap  +  /api/heatmap/reset
│   └── export_csv.py       → /api/export-csv
├── lib/                    ← Shared Python logic
│   ├── classifier.py       ← JudoTechniqueClassifier
│   ├── tracker.py          ← KalmanTrack + MultiPersonTracker
│   ├── heatmap_tracker.py  ← ContactHeatmapTracker
│   └── state.py            ← Shared singleton instances
├── public/                 ← Static frontend (served by Vercel)
│   ├── index.html          ← Login page  (copy from notebook + URL rewrites)
│   └── app.html            ← Analyzer app (copy from notebook + URL rewrites)
├── app.py                  ← Local Flask dev server (NOT deployed)
├── vercel.json             ← Vercel routing config
├── requirements.txt        ← Python dependencies
├── .env.local              ← Local secrets (never commit)
└── .gitignore
```

---

## Step 1 — Extract the HTML from the Notebook

Open `DLSU_judo_analyzer_2_.ipynb` and find cell 19 which contains two Python
string variables: `LOGIN_HTML` and `HTML_PAGE`.

1. Copy the **content** of `LOGIN_HTML` (everything between the triple quotes)
   into `public/index.html`

2. Copy the **content** of `HTML_PAGE` into `public/app.html`

---

## Step 2 — Rewrite the API URLs in the HTML

These `fetch()` calls need updating in both HTML files.

### In `public/index.html` (the login page JS):
```
fetch('/login',    ...)  →  fetch('/api/auth?action=login', ...)
fetch('/register', ...)  →  fetch('/api/auth?action=register', ...)
```

### In `public/app.html` (the analyzer app JS):
```
fetch('/analyze', ...)        →  fetch('/api/analyze', ...)
fetch('/export-csv')          →  fetch('/api/export-csv')
fetch('/heatmap')             →  fetch('/api/heatmap')
fetch('/reset-heatmap', ...)  →  fetch('/api/heatmap/reset', ...)
fetch('/me')                  →  fetch('/api/auth?action=me')
href="/logout"                →  href="/api/auth?action=logout"
```

### Add auth guard to `public/app.html`

Add this at the very top of the inline `<script>` block in app.html:

```js
// Redirect to login if not authenticated
(async () => {
  const r = await fetch('/api/auth?action=me');
  if (!r.ok) window.location.href = '/';
})();
```

---

## Step 3 — Local Setup

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free tier is fine)

### Install dependencies
```bash
cd judo-analyzer
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install flask               # only needed for local dev server
```

### Configure environment
```bash
cp .env.local .env.local        # already exists — just fill in values
```

Edit `.env.local`:
```
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
```

### Run locally
```bash
python app.py
```
Open http://localhost:5000

---

## Step 4 — Deploy to Vercel

### Prerequisites
- [Vercel account](https://vercel.com) (free)
- [Vercel CLI](https://vercel.com/docs/cli): `npm i -g vercel`
- Project pushed to a GitHub/GitLab repo

### One-time setup
```bash
cd judo-analyzer
vercel login
vercel link          # links local folder to a Vercel project
```

### Set environment variables on Vercel
```bash
vercel env add MONGO_URI
# paste your MongoDB Atlas URI when prompted

vercel env add SECRET_KEY
# paste a random secret key
```

Or go to: Vercel Dashboard → Your Project → Settings → Environment Variables

### Deploy
```bash
vercel --prod
```

Your app will be live at `https://your-project-name.vercel.app`

---

## ⚠️ Important Limitation — Serverless State

The Python classifier and heatmap tracker keep state **in memory** within a
single serverless function invocation (warm container).

**What this means in practice:**
- While analysing a video in one browser session, state accumulates normally
  because Vercel reuses the warm container for that session.
- If the container is recycled (cold start) mid-session, the event log and
  heatmap will reset.
- Two users analysing simultaneously will get separate containers and separate
  state — which is actually correct behaviour.

**For production persistence**, move the event log writes and heatmap data into
MongoDB. The `db` connection is already set up in `api/auth.py` — you can
import it in `api/analyze.py` the same way.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML/CSS/JS (static, no framework) |
| Pose detection | TensorFlow.js + MoveNet MultiPose (client-side) |
| API | Python serverless functions on Vercel |
| Auth | itsdangerous signed cookies + bcrypt |
| Database | MongoDB Atlas |
| Analysis | NumPy, SciPy, Pandas (Kalman filter, technique classifier) |
