# Veris 🏥
### AI-Powered Clinical Decision Support System for Small Hospitals

Veris is a full-stack Django web application that unifies three problems small hospitals typically have no solution for: patient records management, AI-assisted differential diagnosis from symptoms, and automated lumbar spine MRI grading. All three modules live under one roof, with AI embedded from the start rather than bolted on later.

---

## The Three Modules at a Glance

| Module | What it does | AI component |
|---|---|---|
| **Patient Management** | Two-step registration, medical history, auto-generated MRN | None — pure clinical data |
| **Differential Diagnosis** | Symptom-to-disease matching, ranked differential | Feedforward network + Monte Carlo Dropout + hybrid clinical scorer |
| **Spine MRI Grading** | Upload DICOM, get severity predictions for 25 spine conditions | 4-model pipeline: ResNet-18 keypoint regression + 3× ResNet-34 severity classifiers |

---

## Features

**Authentication & Roles**
- Custom `AbstractBaseUser` with two roles: `ADMIN` and `DOCTOR`
- Admins manage the hospital — register doctors, register patients, assign visits
- Doctors log in via a separate flow with TOTP OTP verification and a forced first-login password change

**Patient Management**
- Two-step registration form: personal info → medical history
- Auto-generated patient ID in format `PT{DDMMYYYY}{NNNNN}` (e.g. `PT2303202600001`), sequential per date, never reused
- Age and BMI stored at registration time (historical snapshot, not dynamic computation)
- Medical history stored as `PatientMedicalHistory` with one row per visit, `JSONField` for variable-length medication and condition lists
- Three consent checkboxes: accuracy declaration (mandatory), data policy, optional research consent

**Doctor Module**
- Auto-generated doctor ID (`D-0001`, `D-0002`, ...)
- Department, shift timings, chamber number, schedule
- M2M assignment of patients to doctors
- OTP secret stored per doctor for login verification

**Visit & Queue Management**
- Today's queue with real-time status tracking: `waiting → with_doctor → done`
- Visits linked to both patient and doctor; each visit stores diagnosis notes, AI differential, entered symptoms, selected diagnosis, medications, and next appointment date
- AI diagnosis results saved back to the visit record via a dedicated AJAX endpoint

**Differential Diagnosis — Module 2 (symcat)**
- Input: up to 377 binary symptoms (present/absent)
- Architecture: `377 → 1024 → 512 → 256 → 773` feedforward network with BatchNorm and Dropout (0.4 → 0.3 → 0.2)
- Training data: Kaggle patient case dataset (246,945 rows, after removing 23.2% duplicates) merged with SymCAT clinical weights (801 diseases)
- Inference: 50-pass Monte Carlo Dropout gives mean probability + uncertainty (standard deviation) per disease
- Hybrid scorer re-ranks top predictions using 5 weighted signals: symptom completeness (35%), symptom coverage (25%), neural network probability (20%), specificity (15%), critical symptom penalty (5%)
- Output: ranked differential with confidence intervals, matched/missing symptoms with SymCAT weight categories, and an auto-generated interpretation sentence per disease
- Top-3 accuracy: 94.39% on test set

**Spine MRI Grading — Module 3 (rsna)**
- Upload a DICOM ZIP or folder via the browser
- Classifies severity (Normal/Mild, Moderate, Severe) for 5 conditions across 5 disc levels — 25 predictions per patient
- Conditions: Spinal Canal Stenosis, Left/Right Neural Foraminal Narrowing, Left/Right Subarticular Stenosis
- DICOM geometry used to cross-reference sagittal and axial scan planes (uses `ImagePositionPatient` and `ImageOrientationPatient` tags — not unreliable `InstanceNumber` ordering)
- Test-time augmentation: 4 passes (original, H-flip, ±5° rotation), probabilities averaged
- Temperature scaling applied post-training to calibrate overconfident softmax outputs
- Model 1 validation result: 13.75 px mean pixel error, 80.8% of predictions within 20 px

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x, Django 5.2 |
| Database | SQLite (dev) |
| AI — Diagnosis | PyTorch feedforward network |
| AI — MRI Grading | PyTorch, ResNet-18, ResNet-34 |
| Image processing | pydicom, OpenCV, NumPy, SciPy |
| Frontend | Django templates, HTML/CSS/JS |
| Session cache | Django `LocMemCache` |
| Auth | Custom `AbstractBaseUser` + TOTP OTP |

---

## Project Structure

```
veris/
├── manage.py
├── requirements.txt
├── .env                        # Local secrets — never committed
├── .gitignore
│
├── src/                        # Training scripts for Imaging module
├── healthcare/                 # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── accounts/                   # Auth: custom User model, login, admin home dashboard
│   ├── models.py               # AbstractBaseUser with ADMIN / DOCTOR roles
│   ├── views.py                # LoginView, home_view (admin dashboard with 7-day visit chart)
│   └── urls.py
│
├── patients/                   # Patient registration and medical history
│   ├── models.py               # Patient, PatientMedicalHistory
│   ├── views.py                # Two-step registration form + AJAX patient search
│   └── urls.py
│
├── doctors/                    # Doctor profiles and scheduling
│   ├── models.py               # Doctor (linked 1:1 to User, M2M to Patient)
│   ├── views.py                # Registration, OTP flow, dashboard, patient queue JSON
│   └── urls.py
│
├── visits/                     # Visit queue and clinical notes
│   ├── models.py               # Visit (patient + doctor + status + AI fields)
│   ├── views.py                # Queue, add/update visit, mark done, save AI result
│   └── urls.py
│
├── symcat/                     # Differential diagnosis AI module
│   ├── views.py                # predict_disease endpoint + symptom checker page
│   ├── models/
│   │   └── symptom_disease_best.pt   # Trained PyTorch model (15 MB)
│   └── data/
│       ├── merged_symptom_disease_data.csv
│       ├── symptom_encoder.pkl
│       └── idx_to_disease.pkl
│
├── rsna/                       # Lumbar spine MRI grading module (RSNA 2024)
│   ├── apps.py                 # Loads all 4 ML models at server startup via ready()
│   ├── views.py                # DICOM upload handler
│   ├── services.py             # process_dicom_folder — orchestrates full inference pipeline
│   ├── urls.py
│   ├── config/
│   │   └── settings.py         # CFG: all paths and hyperparameters in one place
│   ├── src/
│   │   ├── module2.py          # DICOM loading, windowing, 3-slice patch extraction (.npy)
│   │   ├── module3.py          # PyTorch Dataset, DataLoader, augmentation pipeline
│   │   ├── module4_model1.py   # ResNet-18 keypoint regression (training script)
│   │   ├── module5_inference.py # Full inference: keypoints → patches → severity grades
│   │   └── logger.py
│   ├── checkpoints/            # Trained model weights (download separately — not in repo)
│   │   ├── model1_best.pt      # Keypoint regression
│   │   ├── model2_best.pt      # SCS classifier
│   │   ├── model3_best.pt      # NFN classifier (or model3_left/right_best.pt)
│   │   ├── model4_best.pt      # SS classifier
│   │   ├── nfn_keypoint_offsets.json
│   │   ├── ss_axial_offsets.json
│   │   └── temperature_scales.json
│   ├── data/raw/               # RSNA CSVs and DICOM images (not in repo)
│   └── outputs/logs/
│
├── static/                     # CSS, JS, images
├── templates/                  # Shared HTML templates
└── media/                      # User uploads (patient photos, DICOM study uploads)
    └── rsna_uploads/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/veris.git
cd veris
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install python-dotenv
```

### 4. Configure environment variables

Create a `.env` file in the project root (next to `manage.py`):

```
SECRET_KEY=your-django-secret-key-here
DEBUG=True
```

> Generate a new secret key: `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`

### 5. Apply migrations and create a superuser

```bash
python manage.py migrate
python manage.py createsuperuser
```

The superuser email becomes the Admin login. Use it to register doctors from the admin dashboard.

### 6. Place model checkpoints

The RSNA model weights are too large for Git. Place them in `rsna/checkpoints/` after creation using instructions stated later in this summary:

| File | Description |
|---|---|
| `model1_best.pt` | ResNet-18 keypoint regression |
| `model2_best.pt` | ResNet-34 SCS severity classifier |
| `model3_best.pt` | ResNet-34 NFN severity classifier |
| `model4_best.pt` | ResNet-34 SS severity classifier |
| `nfn_keypoint_offsets.json` | Pre-computed T2→T1 coordinate offsets |
| `ss_axial_offsets.json` | Pre-computed axial slice offsets |
| `temperature_scales.json` | Calibration scalars per model |

The diagnosis model (`symcat/models/symptom_disease_best.pt`) is included in the repo at 15 MB.

### 7. Run the development server

```bash
python manage.py runserver
```

Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — it redirects to `/login/`.

> **Note:** The RSNA models load at server startup via `rsna/apps.py → ready()`. If no checkpoints are present the server starts normally but MRI grading will not function.

---

## URL Reference

| URL | Access | Description |
|---|---|---|
| `/login/` | Public | Admin login |
| `/home/` | Admin | Admin dashboard |
| `/register/` | Admin | Register new patient (Step 1) |
| `/medical-history/<id>/` | Admin | Patient medical history (Step 2) |
| `/doctors/register/` | Admin | Register a new doctor |
| `/doctors/dashboard/` | Doctor | Patient queue and search |
| `/doctors/verify-otp/` | Doctor | TOTP verification on login |
| `/doctors/change-password/` | Doctor | Forced first-login password change |
| `/visits/assign/` | Admin | Check patient in, assign to doctor |
| `/visits/queue/` | Admin | Today's visit queue |
| `/visits/patient/<id>/` | Doctor | Patient detail with visit history |
| `/visits/add/<id>/` | Doctor | Add a visit record |
| `/symptom-checker/` | Doctor | AI differential diagnosis page |
| `/api/symptoms/` | Internal | AJAX — full symptom list |
| `/api/predict/` | Internal | AJAX — run diagnosis inference |
| `/visits/save-ai/<id>/` | Internal | AJAX — save AI result to visit |
| `/rsna/upload/` | Doctor | Upload DICOM study for MRI grading |
| `/rsna/results/<id>/` | Doctor | View spine grading results |
| `/admin/` | Admin | Django admin panel |

---

## How the Diagnosis AI Works

1. Doctor selects symptoms from the full list of 377 binary options.
2. The input vector is passed through the network **50 times** with dropout kept active (Monte Carlo Dropout). Mean of 50 passes = disease probability; standard deviation = uncertainty.
3. A **hybrid scorer** re-ranks the raw top predictions using five clinical signals weighted to prioritise clinical reasoning over raw model output.
4. Top-N ranked diseases are shown with confidence intervals, matched symptoms colour-coded by SymCAT weight, missing key symptoms, and an auto-generated interpretation per disease.
5. Doctor selects a diagnosis; it is saved to the visit record via AJAX.

## How the MRI Grading Pipeline Works

1. Doctor uploads a DICOM ZIP for a patient's lumbar spine study.
2. Files are organised by `StudyInstanceUID / SeriesInstanceUID`; `SeriesDescription` tags identify the three series types.
3. **Model 1 (ResNet-18)** runs on the middle Sagittal T2 slice and predicts normalised (x, y) for all 5 disc levels.
4. **Model 2 (ResNet-34)** crops 3-slice Sagittal T2 patches at keypoints and classifies Spinal Canal Stenosis severity.
5. **Model 3 (ResNet-34)** translates T2 keypoints into Sagittal T1 space using pre-computed offsets, crops patches, classifies Neural Foraminal Narrowing.
6. **DICOM geometry** converts each 2D keypoint to a 3D patient-space coordinate and finds the physically closest Axial T2 slice.
7. **Model 4 (ResNet-34)** crops left and right Axial T2 patches and classifies Subarticular Stenosis.
8. Each classifier runs **4-pass TTA**, probabilities averaged, then divided by a learned temperature scalar.
9. Result: 25 severity predictions displayed on the results page.

---

## Training the RSNA Models (Optional)
The dataset is available at [Kaggle - RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data)

```bash
# 1. Place CSVs and DICOM images under rsna/data/raw/

# 2. Build master index and extract patches
python -m src.module2

# 3. Train Model 1 (must run first — all downstream models depend on its keypoints)
python -m src.module4_model1

# 4. Train Models 2, 3, 4
python -m src.module4_model2
python -m src.module4_model3
python -m src.module4_model4

# 5. Calibrate temperature scaling and run final inference
python -m src.module5_inference --calibrate-only
python -m src.module5_inference

# Smoke test (2 studies only)
python -m src.module5_inference --smoke-only
```

---

## Limitations

- SQLite is used for development. Switch to PostgreSQL for production.
- RSNA model checkpoints must be downloaded separately and placed in `rsna/checkpoints/`.
- The diagnosis model is trained on US-based clinical data — disease prevalence may not reflect Indian epidemiology.
- MRI grading requires Sagittal T2, Sagittal T1, and Axial T2 series with valid DICOM metadata; incomplete or non-standard DICOM may fail.
- Not validated on prospective clinical data. Not approved for clinical use without CDSCO SaMD regulatory clearance.

---

## Future Work

- PostgreSQL + production deployment (Gunicorn + Nginx)
- HL7 FHIR integration for interoperability with existing hospital systems
- Multilingual support (Hindi, Gujarati)
- Indian patient dataset for retraining the diagnosis model
- Mobile-responsive UI and dedicated mobile app
- Prospective clinical validation study

---

## Team

| Name | Enrollment |
|---|---|
| Disha Alagiya | IU2241230503 |
| Ishita Ahir | IU2241230512 |
| Khushi Patel | IU2241230525 |

**Guide:** Asst. Prof. Urvi Rabara, Assistant Professor, CSE Department  
**Institution:** Indus Institute of Technology and Engineering, Indus University, Ahmedabad — Nov 2025

## License

This project was developed as an academic submission for the Bachelor of Technology (Computer Science & Engineering) programme at Indus University. It is intended for educational purposes.

> *Veris* — from Latin *veritas* (truth). Designed to support clinical decision-making with transparent, explainable AI.