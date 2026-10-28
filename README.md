# Outbreak / KNN Laboratory

A functional educational website for the MUJ CCE3131 Disease Outbreak Risk Classification mini-project. It wraps the supplied Python experiment with real server-side KNN inference, CSV validation, persistent experiment tracking and generated figures. The soft blue gradient interface interprets the supplied mood references; those references were not page-by-page application designs.

**Synthetic-data coursework only. Not a medical device, diagnosis, clinical probability estimator, or population outbreak forecasting service.** Never use a Low output to rule out illness. The model's neighbor vote shares are not calibrated clinical probabilities.

## Live website

**Open the app: [https://disease-outbreak-knn-web.onrender.com](https://disease-outbreak-knn-web.onrender.com)**

The full FastAPI/Python application is hosted on **Render Free**, connected to this GitHub repository’s `main` branch with **On Commit** auto-deployment. This is not a static GitHub Pages preview.

- Visitors can use the assessment without an API key. [Try the assessment](https://disease-outbreak-knn-web.onrender.com/assess).
- Training, CSV validation and experiment creation are operator-protected to avoid unrestricted expensive jobs. The repository owner can retrieve `APP_API_TOKEN` from the Render service’s **Environment** settings and enter it in the site’s operator-key field. Never commit or publicly share that key.
- **Free-tier limits:** the service can sleep after 15 minutes without inbound traffic, so the first visit may take about a minute to wake. The local filesystem is temporary: SQLite run history, uploaded CSVs and generated results can be lost on sleep, restart or redeploy. Download any results you need to keep. See [Render Free limits](https://render.com/docs/free).
- Only synthetic or appropriately de-identified coursework data should be used. This is a coursework demo, not clinical or production infrastructure.

Verified live on **28 October 2026**: all five pages, frontend assets, model metadata, real report-example prediction (High; 78.17% neighbor vote share), invalid-input rejection, and anonymous-training rejection. The complete full-data experiment was validated locally; a full search has not been benchmarked on the free host’s limited CPU.

## Features
- Five routes: overview, assessment, experiments, dataset and methodology.
- Real StandardScaler + KNN serving pipeline: seed 42, 80/20 stratified split, k=25, cosine distance, distance weighting.
- Freshly measured overview metrics; the serving preset is explicitly distinguished from a full new grid search.
- Interactive 18-feature assessment. Inputs are not saved.
- Full original experiment as a background job: k=1–40 CV, 90 grid combinations over five folds, raw/scaled baselines, held-out metrics, four plots and downloadable JSON.
- SQLite run history, bounded single-worker queue, failure reporting and isolated run artifacts.
- Synthetic CSV download and validated fixed-schema CSV experiments. Uploaded experiments do not replace the serving model.
- Security headers, bounded requests, strict artifact allowlists and local-only training by default.

## Windows setup (Command Prompt)
Install Python 3.12. Unzip the source and open Command Prompt in this folder.

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Startup fits the synthetic preset model and computes held-out metrics; allow a few seconds. Leave the terminal running. Ctrl+C stops the server. The frontend needs no Node installation or build step.

macOS/Linux equivalent:
```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Configuration
`.env.example` documents configuration; this application reads actual process environment variables, not `.env` files automatically. No external API credentials are needed.

Windows optional settings, before starting:
```bat
set APP_DATA_DIR=C:\path\to\local\outbreak-data
set APP_API_TOKEN=your-locally-generated-long-operator-key
```
Do not paste real secrets into chat or commit them. The optional operator key gates training/upload endpoints. Enter it in the page's operator-key field for that session; it is held in memory only. Prediction remains public if you expose the server, so this key is not a replacement for full deployment authentication.

## Use
1. Assessment → Load report example → Classify profile. Results are computed by sklearn, not hardcoded.
2. Experiments → Run full synthetic experiment. Wait for `completed`, then Inspect. Runs have a 15-minute timeout and queue size of two.
3. Dataset → download synthetic CSV; validate it, confirm coursework data status and launch a CSV experiment.
4. Inspect results, figures, parameters and limitations. KNN CV/grid runs may take several minutes. No automatic promotion to the active model occurs.

## CSV contract
The web requires exactly these columns:
```
fever,cough,fatigue,headache,body_ache,sore_throat,breathlessness,loss_of_smell,diarrhea,vomiting,rash,joint_pain,chills,nausea,age,temperature_c,days_since_onset,contact_with_case,risk
```
100–5,000 rows; at least ten per class; all values finite and numeric except risk labels. Symptoms/contact are 0/1. Age 1–120, temperature 30–45, onset integer 0–60. Labels may be Low/Medium/High or 0/1/2. Upload file limit 1.5 MB; total request limit 2 MB. No missing-value imputation is invented.

For arbitrary numeric feature sets use the retained standalone CLI:
```bat
.venv\Scripts\python research\outbreak_knn.py --csv yourfile.csv --target risk
```
All three classes and at least 100 rows are required. The fixed symptom demo is skipped when a custom schema cannot support it. This CLI does not expose a web prediction form for arbitrary custom features. The original synthetic run is:
```bat
.venv\Scripts\python research\outbreak_knn.py
```
It writes `outputs/` in the current working folder. Changes to source experiment: artifact exports, CV JSON, single-process grid search, pandas 3 string compatibility, numeric/class checks and custom-schema demo guard. The statistical split and search space are retained.

## Storage / privacy
`var/app.sqlite3` tracks run ID, UTC creation time, status, source and error. `var/<id>/` holds uploaded CSV if any, log and output artifacts. Prediction inputs are never persisted. Run artifacts and trained KNN models retain training rows, so use only synthetic or appropriately de-identified coursework data. There is no clinical retention policy or application delete control: stop the server and remove the local `var` folder to reset the entire laboratory. Preserve any results you need first. Unfinished runs become failed on restart. Run with exactly one Uvicorn worker; concurrent server instances sharing a data directory are unsupported.

## Testing
```bat
.venv\Scripts\python -m pytest -q
```
Optional frontend DOM interaction tests (Node 20+):
```bat
npm ci
node tests/frontend.cjs
```
These expect the server on http://127.0.0.1:8765 (or set TEST_BASE_URL). DOM tests are not a substitute for browser rendering/accessibility audits. See `docs/TEST_REPORT.md` and raw QA evidence for exact verified/untested status.

## Deployment
The public deployment above runs the repository Docker image on Render Free (512 MB RAM, 0.1 CPU). The image built and the Python application started successfully on Render. The service has `SKLEARN_WORKING_MEMORY=64`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to bound numerical resource use, plus a generated `APP_API_TOKEN` stored only in Render.

To run the container yourself:
```sh
docker build -t outbreak-knn .
docker run --rm -p 127.0.0.1:8000:8000 -v outbreak-data:/data outbreak-knn
```
The image is verified on Render; execution through a local Docker installation remains untested. A small VM or container host with persistent disk and adequate CPU/RAM is appropriate for coursework; serverless request-limited hosting is a poor fit for background grid search. The hosted demo has HTTPS and operator-only training. Before treating it as production, add full authentication, rate limiting, durable storage, and a data-handling review. Supply APP_API_TOKEN securely and do not trust forwarded client IPs as authentication. Do not expose real patient data. Docker port mapping above intentionally binds the host's loopback interface only.

## Repository handover
Public repository: https://github.com/Madhurjya-K/disease-outbreak-knn-web (main). Initial transfer uses the connected integration because local Git authentication is unavailable. The included local history and remote per-file history differ. For future work, clone the populated remote rather than force-pushing the separate local history.

For a separate empty repository only, the standard commands would be:
```bat
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```
Do not run this blindly against a repository containing unrelated work. To update later, edit, test, inspect `git diff`, commit meaningful changes and push. See `docs/GITHUB_HANDOVER.md` for current status.

## Structure
`app/model.py`: source dataset and serving model; `app/main.py`: validation, API, SQLite and worker; `app/static/`: modular styling and frontend behavior; `app/templates/`: shared shell; `research/`: reproducible experiment; `tests/`: automated checks; `docs/source/`: supplied report; `docs/`: architecture and QA evidence.

## Deliberate non-features
No clinician/admin roles, patient identities, automatic alerts, production authentication, clinical calibration, real-data acquisition, SMOTE, logistic-regression/tree comparisons or regulatory certification. Optional source extensions are documented, not fabricated as completed. The UI is a mood-reference interpretation and can be replaced with detailed screen designs without rewriting inference.
