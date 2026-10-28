# Test and implementation report

Date: 28 October 2026. Environment: Windows, Python 3.12.7, pinned packages in requirements.txt. QA used an isolated Python virtual environment. Node/jsdom is needed only for optional frontend tests.

## Passed
- **35 automated pytest tests**. Raw output: qa/pytest.txt. Coverage includes five page routes, real deterministic inference, all symptom fields, missing/extra fields, invalid values, flag typing, vote-share sum, prediction non-persistence, synthetic download, numeric/text CSV labels, missing/non-finite values, class checks, schema/size constraints, artifact allowlists, origin restriction, operator-key gate, remote training rejection, request-size limit, queue bound, not-ready result handling, run persistence and interrupted-run failure recovery.
- **14 frontend DOM interaction checks** against the actual local API. Raw output: qa/frontend.json. Checked navigation, live overview statistics, all symptoms, report-example preset, real High prediction, three vote shares, reset, displayed server validation error, schema display, real CSV validation, actual CSV job submission, persistent history and methodological notice.
- JavaScript syntax check (`node --check app/static/app.js`). Python imports and serving startup succeeded.
- Local Uvicorn HTTP server served frontend assets, API data and generated images.
- Full 3,000-row synthetic experiment completed through the background worker, not a hardcoded result. Run d5a5d1012d4f42c0abb97c322fd3b180. 80/20 split, k=1..40 five-fold CV and full 90-combination grid search ran. Generated four PNG figures, dataset, CV curve, model artifact and results JSON. Accuracy **0.7883**, macro-F1 **0.7447**, CV best macro-F1 **0.7194**. Selected k=25, cosine, distance weighting. Raw/scaled baselines **0.6650 / 0.7283**. Report-example High vote share **0.782**. These reproduce the supplied report at its rounding precision.
- A separate **200-row synthetic CSV** experiment was submitted through the frontend DOM flow and completed. Run 2ffbff59896f4d1690f7e568111157a5. Accuracy **0.7500**, macro-F1 **0.6903**. This is a small integration fixture, not a claimed performance improvement or clinical result.
- All eight plot endpoints across these two initial runs returned valid PNG bytes; completed result endpoints returned parseable JSON. Evidence: qa/integration.json and qa/live-server.txt.

## Bugs found and fixed
- pandas 3 inferred string dtype did not match the original `dtype == object` condition. Updated text-risk mapping in both web validation and standalone CLI; added regression coverage.
- CSV minimum increased to 100 rows so the k=40 curve is supported inside every training fold.
- Standalone custom-feature CSVs no longer fail solely because the fixed symptom demo lacks their columns; that example is explicitly skipped when incompatible.
- Browser testing caught polling rebuilding unchanged history rows, invalidating click targets and keyboard focus. The frontend now skips identical snapshots and gives Inspect buttons stable run attributes.
- All pytest files share a temporary data root through conftest, protecting actual run artifacts from test cleanup.

## Failed / incomplete / not verified
- No current automated test failures. One non-failing Starlette/httpx test-client deprecation warning remains; this does not affect demonstrated application requests.
- Initial agent browser calls could not attach to the local app tab (`could not be opened for reading` / `failed to wake or finish loading`). API logs confirm the browser requested the shell, CSS, JS and model metadata, but that is not proof of correctly rendered interaction. DOM tests are recorded separately, not called full browser tests.
- Reopening the local app in a fresh agent tab resolved attachment. Actual desktop browser actions verified the report preset, submission, High label and 78.2% vote share. Overview and assessment screenshots are saved under qa/. The completed synthetic experiment Inspect button rendered measured results and four image elements. This is a bounded desktop check, not full responsive certification.
- Tablet/mobile rendering, exact screenshot comparison, keyboard/screen-reader audit, contrast audit and browser console inspection remain unverified. Responsive CSS and accessibility hooks are implemented but not certified by tests.
- Docker build/runtime and public hosting were not executed. A Docker recipe and explicit deployment instructions are included. There is no verified public live URL.
- No real clinical dataset, diagnostic validity, probability calibration, production auth/RBAC, penetration test, multi-instance concurrency, clinical retention controls or load test was provided or claimed.
- Optional source extensions (SMOTE, logistic regression/tree comparison, real-data replacement) are not implemented requirements and remain future experiments.

## Artifact handling
Runtime CSVs, databases, models and logs are excluded from the GitHub source. This report and compact QA evidence are included. The source report and implementation plan are retained under docs/. Generated outputs are available locally under var/<run-id>/outputs. The source ZIP excludes .venv, node_modules and runtime data; set up dependencies using README instructions.

## Final frontend retest
After the polling fix, the controlled live-server driver reran all 14 frontend checks successfully and completed a repeat 200-row CSV experiment (0eb73cf1459a4cf8b39d2b8645b66d38) with the same 0.7500 accuracy and 0.6903 macro-F1. All three persisted completed runs and their 12 plot endpoints were verified. Ad-hoc frontend checks after the temporary shell server ended had connection errors; the controlled server/test lifecycle resolved this harness issue.
