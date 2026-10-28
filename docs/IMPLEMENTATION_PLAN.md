# Disease Outbreak KNN — analysis and implementation plan

## 1. Source of truth and scope
Reviewed: all four pages of Disease_Outbreak_KNN_Project_Report.pdf; the complete outbreak_knn.py; both archive manifests; all five attached visual references; the workflow brief. The nested archive contains the same report/script plus dataset, results and four plots. Original evidence will be preserved separately.

Purpose: turn the MUJ CCE3131 mini-project into an educational web application that classifies an 18-feature symptom vector as Low, Medium or High using KNN. Intended users are students and assessors. The report does not specify hospital operations, clinician accounts, geographic outbreak aggregation, alerts to external parties, or patient records. These will not be invented.

Explicit requirements: reproducible synthetic generator (3,000 rows, seed 42); 14 symptoms plus age, temperature, onset days and contact flag; stratified 80/20 split; training-only scaling; k=1..40 five-fold macro-F1; grid search across odd k=3..31, uniform/distance weights and Euclidean/Manhattan/cosine; raw/scaled baselines; held-out accuracy, macro-F1, class recall and confusion matrix; EDA, CV curve and PCA visualization; new-vector prediction; CSV input; clear limitations; runnable code and results.

Optional source extensions, not core requirements: logistic-regression/tree comparison, SMOTE, and acquisition of a real clinical dataset. These will remain explicitly optional. No clinical validation is implied by the provided accuracy. KNN vote shares are not calibrated medical probabilities. Individual symptom classification does not establish population-level outbreak risk.

## 2. Proposed stack and rationale
Python 3.11+ with FastAPI/Uvicorn keeps inference and training in the same language as the source. scikit-learn, pandas, NumPy, matplotlib and seaborn retain the original experiment. HTML/CSS and modular browser JavaScript provide a small responsive frontend without a Node build step. SQLite (standard library) stores experiment status and metadata; local filesystem stores synthetic datasets, plots and trained artifacts. No paid APIs or external credentials are required.

No authentication is required by the source. Default deployment binds to 127.0.0.1 and is single-user educational software. Before public multi-user deployment, add an authenticated reverse proxy, TLS, rate limits and operator-only training access; this initial application must not be represented as public clinical infrastructure. Prediction inputs are not persisted. Uploaded CSVs must be synthetic or de-identified coursework data; training artifacts retain rows and are sensitive if real data is supplied.

## 3. Pages and feature contracts
| Page/module | Behavior and components | Verification |
|---|---|---|
| Overview | Active model provenance, class/sample counts, measured metrics, navigation and warning | Data comes from backend metadata, not copied report values |
| Assessment | 14 labeled symptom toggles, age/temperature/onset/contact inputs, demo preset, submit/reset; class and vote shares | POST validated inputs; values sum to one; no automatic external action; invalid/missing values rejected |
| Experiments | Launch complete original experiment, persistent status, run list, metrics, parameters, four generated plots, result download | Training occurs outside request handler; completion only after outputs verified; failed runs show error |
| Dataset | Explain synthetic generator and exact feature schema; download active synthetic data; validate and launch fixed-schema CSV experiments | Reject wrong columns, non-finite values, invalid labels/flags/ranges, excessive size, too few rows/class |
| Methodology | KNN/scaling/split/CV explanation, source limitations, PCA caveat and viva material | All source method and limitation topics accessible without claiming clinical validity |

CSV compatibility: original CLI remains available for arbitrary numeric feature sets. The web form deliberately supports only the documented 18-feature schema so its assessment fields cannot silently mismatch a custom model. CSV experiments are evaluated separately and do not replace the active synthetic demo model. This is a documented safety/usability choice.

## 4. Architecture and data flow
Browser pages -> same-origin JSON API -> validation/model service -> sklearn pipeline. Assessment -> validated DataFrame in fixed feature order -> StandardScaler -> KNN -> label/vote shares. The initial serving model is freshly fitted using the source dataset and reported preset (k=25, distance, cosine); its held-out metrics are recalculated and marked preset, not a newly completed grid search.

Experiment POST -> SQLite queued record -> one background worker -> original script in isolated per-run working directory -> real CV/grid search -> plots/results/model -> SQLite completed or failed. Subprocess limits bound execution; only a single worker processes jobs. Interruptions on restart are recorded as failed. CSV request bytes are size-limited and parsed/validated before enqueue. SQLite metadata and run output directories persist across restarts.

API: GET /api/health; GET /api/model; POST /api/predict; GET/POST /api/runs; GET /api/runs/{id}; GET /api/runs/{id}/results; GET /api/runs/{id}/plots/{name}; GET /api/dataset; POST /api/dataset/validate. CSV experiment creation uses a JSON csv_text field, avoiding multipart dependency. Pages /, /assess, /experiments, /dataset, /methodology use a shared shell and separate render logic.

Schema: experiment_runs(id primary key UUID hex, created_at UTC, status queued/running/completed/failed, source synthetic/upload, error nullable). Filesystem directory keyed by run ID holds input.csv (if supplied), log.txt, outputs/results.json, figures, dataset_used.csv, model.joblib and CV curve. SQLite contains no patient identity table. Each artifact belongs to one experiment. No prediction history is retained.

## 5. Layout and visual interpretation
Images are mood references, not exact application screen specifications. Use pale blue/white surfaces, diffuse cobalt gradients, clear dark text, generous negative space, restrained serif display type and practical sans-serif controls. Create gradients in CSS rather than embedding unrelated trading text, watermarks or copied branding. Keep tokens in CSS and rendering separate from API/model logic. Adapt grids to narrow screens; labeled fields, keyboard focus, reduced-motion support, adequate contrast and live status regions.

## 6. Folder structure
app/main.py (API, validation, persistence and job orchestration); app/model.py (synthetic serving model); app/static/ (CSS/JS); app/templates/index.html; research/outbreak_knn.py (original experiment plus artifact export and single-process adaptation); tests/; docs/; requirements.txt; .env.example; .gitignore; Dockerfile; README.md. Runtime data under var/ is ignored. Original source report is retained in docs/source/ for provenance.

## 7. Security, errors and operational constraints
Never load uploaded pickle/joblib. Only load app-generated artifacts. Do not render CSV or server errors as HTML. Use textContent for dynamic output, strict path allowlists for artifacts, UUID validation, bounded CSV size/row count and bounded prediction values. Do not commit runtime inputs, databases, model binaries or secrets. No CORS wildcard. Set browser security headers. Training APIs are local-only unless an operator configures an API token; public deployment needs additional identity/rate-limit controls. Report explicit errors and empty/loading/success states. No fake live numbers or model confidence claims.

## 8. Milestones and acceptance criteria
1. Analysis/plan: complete source inventory, requirement/assumption separation, architecture and risks (this document).
2. Core: install dependencies; implement real serving pipeline, validation, storage and worker. Acceptance: valid predictions and actionable invalid-input errors; no input persistence.
3. Web: implement all five routes and responsive reference-inspired components. Acceptance: working navigation/forms/downloads/status updates, keyboard labels and synthetic-data warnings.
4. Experiment: execute complete source-equivalent CV/grid search. Acceptance: actual metrics, all four generated plots, terminal run status and downloadable JSON; explicitly distinguish measured outputs from supplied report.
5. QA: automated API/model/CSV/storage/security tests, live server requests and browser workflow checks; separately record unsupported checks. Acceptance: report exact pass/fail/untested status, not assumptions.
6. GitHub: confirm owner/repository/branch and visibility, inspect destination, commit organized source with secret exclusions, push and independently read back commit/files. Connector has file read/write actions but no repository-create action; use browser creation if needed and authorized. Never overwrite unrelated work.
7. Handover: source ZIP, setup guide, environment template, deployment recipe and test report. Local Docker/FastAPI deployment readiness is not a verified public deployment.

## 9. Deployment and setup
Windows: create .venv, install requirements, run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000. SQLite and artifacts use APP_DATA_DIR. Docker provides a repeatable single-process deployment; mount /data persistently and set APP_DATA_DIR=/data. Training assumes one Uvicorn worker. No cloud deployment or recurring paid service will be created without a confirmed target/account.

## 10. Open decisions / limits
GitHub target owner, repository name, branch and visibility must be confirmed before external writes. A later page-specific UI can replace the modular stylesheet/shell. Real clinical performance, regulatory suitability and calibrated probability estimation are out of scope and unverified. Source methodological caveat: raw/scaled baseline test results are inspected alongside tuned results; reserve a genuinely external validation set for future scientific claims. PCA refits a separate 2-D visualization classifier, not the deployed 18-D model.
