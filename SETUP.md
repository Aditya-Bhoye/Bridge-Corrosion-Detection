# BridgeGuard dashboard

## Frontend

Requires Node.js 20.19+ or 22.12+.

```powershell
cd E:\Bridge-Corrosion-Detection
npm install
npm run dev
```

Open http://127.0.0.1:5173. Demo mode is available without Python. Demo mode always returns the same illustrative 18.7% coverage and does not analyze images.

## Real image detection

Use Python 3.11 or 3.12 with compatible PyTorch wheels. In another terminal:

```powershell
cd E:\Bridge-Corrosion-Detection
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Export `corrosion_unet_enhanced.pth` by running the existing training notebook and place it in `backend/`. Alternatively set `$env:MODEL_PATH` to the checkpoint's full path before starting the API. The API loads the exact notebook architecture and enhancement pipeline on CPU. No weights or dataset are supplied in this repository. Model loading errors are shown on analysis; the health check reports checkpoint file presence, not checkpoint validity.

Select **Trained U-Net model** in the dashboard. Vite forwards `/api` to port 8000. The upload is processed in memory. Masks use a prediction threshold of 0.5 and are displayed over the original image. History stores only filenames, dates, source and coverage in this browser, up to 50 records; original photos and masks are not persisted. History can be exported as JSON or individual records deleted.

Coverage bands are UI heuristics: low <10%, moderate 10% to <25%, high ≥25%. These describe the fraction of image pixels classified as corrosion, not bridge structural severity or engineering safety. Model confidence is not reported because the notebook provides no calibrated confidence metric.

## Verification and deployment

```powershell
npm test
npm run build
```

`dist/` contains the production frontend. A production deployment must serve `/api` from the Python service on the same origin (the Vite development proxy is not part of the build). The API is intended for trusted local use; add authentication and operational safeguards before exposing it publicly.
