# Judo Analyzer — Project Documentation

This document provides a comprehensive overview of the **Judo Analyzer** project architecture, key components, and developer guidance.

---

## Project Overview

**Judo Analyzer** is a web-based system for analyzing judo matches. It uses computer vision (pose estimation) to track athletes, classify techniques (grips, throws, pins), and generate heatmaps of contact points. The application was originally built as a Google Colab notebook and has been migrated to a proper Vercel deployment with a Python backend and web frontend.

**Key Features:**
- User authentication (login/register)
- Real-time judo technique classification
- Multi-person athlete tracking (Kalman filter)
- Contact heatmap generation
- CSV data export
- Video analysis with frame-by-frame pose data

---

## Project Structure

```
judo-analyzer/
├── api/                          # Vercel serverless Python functions
│   ├── index.py                  # Main Flask app (single-file deployment)
│   ├── auth.py                   # Separate auth logic (importable)
│   ├── analyze.py                # Separate analyze logic (importable)
│   ├── heatmap.py                # Heatmap logic (importable)
│   ├── export_csv.py             # CSV export logic (importable)
│   └── lib/                       # Shared Python utilities
│       ├── __init__.py
│       ├── classifier.py         # JudoTechniqueClassifier class
│       ├── tracker.py            # KalmanTrack & MultiPersonTracker classes
│       ├── heatmap_tracker.py    # ContactHeatmapTracker class
│       └── state.py              # Singleton instances (classifier, heatmap_tracker)
├── public/                        # Static frontend (served by Vercel)
│   ├── index.html                # Login page
│   ├── app.html                  # Main analyzer app (JavaScript UI)
│   ├── manage.html               # Management dashboard
│   └── sample/                   # Sample videos/data for demo
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel routing & build config
├── .env.local                    # Local environment variables (not committed)
├── .gitignore
└── README.md                     # Setup & deployment guide
```

---

## Architecture

### Deployment Targets

1. **Local Development**
   - Flask dev server (port 5000 by default)
   - Import from `api/` and `lib/` directories
   - Used with `app.py` or manual Flask setup

2. **Vercel Production**
   - Serverless Python backend via `@vercel/python`
   - All logic inlined into `api/index.py` (no cross-directory imports on serverless)
   - Static frontend via `@vercel/static`
   - URL routing via `vercel.json`

### API Endpoint Routing

| Endpoint | Method | Handler | Purpose |
|----------|--------|---------|---------|
| `/api/auth?action=login` | POST | `auth.py` | User login |
| `/api/auth?action=register` | POST | `auth.py` | User registration |
| `/api/auth?action=logout` | GET | `auth.py` | Clear session |
| `/api/auth?action=me` | GET | `auth.py` | Get current user |
| `/api/analyze` | POST | `analyze.py` | Classify techniques + track athletes |
| `/api/heatmap` | POST | `heatmap.py` | Get contact heatmap |
| `/api/heatmap/reset` | POST | `heatmap.py` | Reset heatmap state |
| `/api/export-csv` | POST | `export_csv.py` | Export session data to CSV |
| `/public/app.html` | GET | (static) | Main analyzer UI |
| `/public/index.html` | GET | (static) | Login page UI |
| `/public/manage.html` | GET | (static) | Management dashboard |

---

## Key Components

### 1. **Authentication** (`api/auth.py`)

**Functionality:**
- User login/register with password hashing (bcrypt)
- Session tokens using `itsdangerous.URLSafeSerializer`
- MongoDB backend for user storage

**Key Functions:**
- `login()` — Validate credentials, create session token
- `register()` — Hash password, store user in MongoDB
- `logout()` — Clear session cookie
- `get_current_user()` — Extract user from session token
- `read_token()` — Deserialize session cookie
- `make_token()` — Serialize session payload

### 2. **Analysis Engine** (`api/analyze.py`)

**Functionality:**
- Accepts pose estimation data (keypoints per athlete)
- Classifies judo techniques (grips, throws, balance disruption)
- Tracks multiple athletes simultaneously
- Returns technique classification + confidence scores

**Input Format:**
```json
{
  "poses": {
    "0": { "0": {"x": 100, "y": 200}, "1": {...}, ... },
    "1": { "0": {...}, ... }
  },
  "videoTime": 123.45
}
```

**Output Format:**
```json
{
  "athletes": {
    "0": {
      "grip": "...",
      "grip_confidence": 0.95,
      "throw_attempted": true,
      "technique_name": "O Goshi",
      ...
    },
    "1": {...}
  },
  "interaction": true,
  "interaction_confidence": 0.87
}
```

### 3. **Classifier** (`lib/classifier.py` / `api/analyze.py`)

**Class: `JudoTechniqueClassifier`**

**Purpose:** Detects and classifies judo techniques from pose keypoints.

**Key Methods:**
- `analyze(poses, video_time)` — Main classification pipeline
  - Detects grips (sleeve, collar, lapel, pant, cross)
  - Analyzes balance disruption (COM displacement)
  - Classifies throws (O Goshi, Seoi Nage, Uchi Mata, etc.)
  - Returns technique confidence scores

**Detection Logic:**
- Uses joint angle calculations between keypoints (e.g., elbow, shoulder, hip)
- Tracks historical data (60-frame window) to detect state changes
- Compares center-of-mass (COM) displacement between frames
- Cross-references opponent keypoints for interaction detection

**Keypoint Mapping** (from pose model):
- 0: Nose, 1: L-Eye, 2: R-Eye, ..., 5: L-Shoulder, 6: R-Shoulder, 11: L-Hip, 12: R-Hip, ...
- Arm angles: calculated from shoulder → elbow → wrist joints
- Core balance: COM of `[5, 6, 11, 12]` (shoulders + hips)

### 4. **Tracker** (`lib/tracker.py` / `api/analyze.py`)

**Class: `KalmanTrack`**

**Purpose:** Kalman-filter-based tracking of a single athlete's center-of-mass (x, y, vx, vy).

**Key Methods:**
- `predict()` — Update state estimate (motion model)
- `update(measurement)` — Correct state with observed position
- `position` (property) — Return current (x, y) position

**Class: `MultiPersonTracker`**

**Purpose:** Manage tracking of multiple athletes, associating detections to tracks.

**Key Methods:**
- `update(detections)` — Match new detections to existing tracks
- `get_tracks()` — Return all active tracks
- Uses Hungarian algorithm for optimal assignment (missing from `lib/tracker.py` — may be in merged `index.py`)

**Filter Design:**
- State: `[x, y, vx, vy]`
- Motion model (F): Constant velocity
- Process noise (Q): Low (assumes smooth motion)
- Measurement noise (R): Low-moderate (pose estimation is fairly accurate)

### 5. **Heatmap Tracker** (`lib/heatmap_tracker.py` / `api/heatmap.py`)

**Class: `ContactHeatmapTracker`**

**Purpose:** Accumulate contact points over a match and generate 2D heatmaps.

**Key Methods:**
- `add_contact(athlete_id, x, y)` — Record contact point
- `get_heatmap()` — Return accumulated heatmap (2D array or image)
- `reset()` — Clear accumulated data

**Heatmap Output:** Usually a 2D array (grid-based) or annotated image showing frequency of contact at each pixel location.

### 6. **Global State** (`lib/state.py`)

**Purpose:** Singleton instances shared across requests.

**Instances:**
- `classifier` — Shared `JudoTechniqueClassifier` (tracks historical data)
- `heatmap_tracker` — Shared `ContactHeatmapTracker`

**Note:** State is NOT reset between API calls, allowing accumulation across frames in a single analysis session.

---

## Frontend Architecture

### HTMLPage Structure

**`public/index.html` — Login Page**
- User login form
- User registration form
- Uses fetch to POST to `/api/auth?action=login` and `/api/auth?action=register`
- Redirects to `/app` on successful login

**`public/app.html` — Main Analyzer**
- Video player or frame upload
- Real-time pose visualization (canvas overlay)
- Technique classification display
- Heatmap visualization
- Session data export button
- Auth guard at top (redirects to `/` if not authenticated)

**Key Fetch Calls in `app.html`:**
```javascript
// Check authentication
fetch('/api/auth?action=me')

// Analyze frame
fetch('/api/analyze', { method: 'POST', body: JSON.stringify({poses, videoTime}) })

// Get heatmap
fetch('/api/heatmap', { method: 'POST', body: JSON.stringify({...}) })

// Reset heatmap
fetch('/api/heatmap/reset', { method: 'POST' })

// Export data
fetch('/api/export-csv', { method: 'POST', body: JSON.stringify({data}) })

// Logout
href="/api/auth?action=logout"
```

---

## Dependencies

**Backend (`requirements.txt`):**
- `flask>=3.0.0` — Web framework
- `numpy>=1.24.0` — Numerical computing
- `pandas>=2.0.0` — Data manipulation
- `scipy>=1.10.0` — Scientific algorithms (distance, etc.)
- `scikit-learn>=1.2.0` — Machine learning (optional, may not be used)
- `flask-bcrypt>=1.0.1` — Password hashing
- `itsdangerous>=2.1.2` — Session token serialization
- `pymongo[srv]>=4.5.0` — MongoDB connection

**Frontend:**
- Vanilla JavaScript (no framework)
- Canvas API for visualization
- Fetch API for HTTP requests

---

## Environment Variables

**`.env.local` (not committed — fill in your own values):**

```bash
# Flask/Vercel
SECRET_KEY=your-secret-key-here

# MongoDB Atlas
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/

# Optional: local dev settings
FLASK_ENV=development
FLASK_DEBUG=1
```

---

## Deployment

### Local Development

1. **Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install flask  # for local dev only
   ```

2. **Configure `.env.local`** with `SECRET_KEY` and `MONGO_URI`.

3. **Run:** 
   - Use `app.py` (if it exists) or create a simple Flask runner
   - Or use Vercel locally: `vercel dev`

### Vercel Deployment

1. **Connect GitHub repo** to Vercel (auto-deploys on push to main)
2. **Set environment variables** in Vercel dashboard:
   - `SECRET_KEY`
   - `MONGO_URI`
3. **Deploy:** Push to GitHub; Vercel auto-builds and deploys

**Build Process:**
- `vercel.json` specifies builds for `api/index.py` (Python) and `public/**` (static)
- Routes requests via `vercel.json` routing rules

---

## Data Flow

### Typical Analysis Session

```
User Upload Frame
    ↓
Pose Estimation Model (external, not included)
    ↓
Frame Keypoints JSON → POST /api/analyze
    ↓
[JudoTechniqueClassifier.analyze()]
  ├─ Compute grip state
  ├─ Compute throw attempt
  ├─ Look up technique name
  └─ Return classifications
    ↓
Frontend receives technique classifications
    ↓
[Optional] POST /api/heatmap (accumulate contact points)
    ↓
[Optional] POST /api/export-csv (export accumulated data)
```

### Session Persistence

- **Global State:** `classifier` and `heatmap_tracker` persist across requests
- **User Session:** Stored in signed cookie (`judo_session`), validated on each request
- **Match Data:** Accumulated in memory; should be reset between matches or saved to MongoDB

---

## Development Notes

### Common Tasks

1. **Add a new technique classification:**
   - Edit `lib/classifier.py` → `_technique_name()` method
   - Add detection logic to `analyze()` or helper methods

2. **Modify heatmap behavior:**
   - Edit `lib/heatmap_tracker.py` → `ContactHeatmapTracker` class
   - Control grid resolution, color mapping, etc.

3. **Add API endpoint:**
   - Add route to `api/index.py` (or modularize in `api/new_feature.py`)
   - Update `vercel.json` if needed (usually not — wildcard routing handles it)
   - Update frontend `fetch()` calls

4. **Deploy:**
   - Commit to GitHub
   - Vercel auto-deploys via webhook

### Testing

- Local: Use `flask` dev server or simulator scripts
- Pose data: Can be mocked JSON in test files
- MongoDB: Use MongoDB Atlas free tier for testing

---

## Future Enhancements

- [ ] WebSocket support for real-time multi-user analysis
- [ ] Video processing pipeline (extract frames, run pose model)
- [ ] Advanced ML models for technique prediction
- [ ] Historical match statistics and analytics dashboard
- [ ] Role-based access control (coach, referee, viewer)
- [ ] Persistent match history in MongoDB

---

## Troubleshooting

### Vercel Deployment Issues

**Problem:** `ModuleNotFoundError` when importing from `lib/`
- **Cause:** Vercel replaces import paths; `api/index.py` must inline all logic
- **Solution:** Copy-paste code from `api/auth.py`, `analyze.py`, etc. into `index.py`, or use `sys.path` manipulation (see `api/analyze.py`)

**Problem:** Session token not persisting
- **Cause:** Cookie scope/domain mismatch
- **Solution:** Ensure `SECRET_KEY` is consistent; set cookie path to `/`

### MongoDB Connection

**Problem:** `MONGO_URI` not recognized
- **Cause:** Environment variable not set in Vercel dashboard
- **Solution:** Add to Vercel project settings; redeploy

**Problem:** Connection timeout
- **Cause:** IP allowlist not configured in MongoDB Atlas
- **Solution:** In Atlas dashboard, add `0.0.0.0/0` to IP allowlist (for development only)

---

## File Checklist

- [x] `README.md` — Deployment guide
- [x] `requirements.txt` — Dependencies
- [x] `api/index.py` — Main Flask app
- [x] `api/auth.py` — Auth logic
- [x] `api/analyze.py` — Analysis logic
- [x] `api/heatmap.py` — Heatmap logic
- [x] `api/export_csv.py` — Export logic
- [x] `lib/classifier.py` — Classifier class
- [x] `lib/tracker.py` — Tracker classes
- [x] `lib/heatmap_tracker.py` — Heatmap tracker class
- [x] `lib/state.py` — Global state
- [x] `public/index.html` — Login HTML/JS
- [x] `public/app.html` — Analyzer HTML/JS
- [x] `public/manage.html` — Management dashboard (optional)
- [x] `vercel.json` — Vercel config
- [x] `.env.local` — Env vars (local only, not committed)
- [x] `.gitignore` — Git ignore rules

---

## Links & References

- **Vercel Python Docs:** https://vercel.com/docs/serverless-functions/python
- **MongoDB Python Driver:** https://pymongo.readthedocs.io/
- **Flask Documentation:** https://flask.palletsprojects.com/
- **Kalman Filter Intro:** https://en.wikipedia.org/wiki/Kalman_filter
- **MediaPipe Pose (implied pose model):** https://developers.google.com/mediapipe/solutions/vision/pose_estimation

---

*Last Updated: 2025 | Migrated from Google Colab notebook → Vercel deployment*
