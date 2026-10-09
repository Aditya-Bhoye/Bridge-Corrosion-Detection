# BridgeGuard dashboard

BridgeGuard runs the trained U-Net **inside the visitor's browser**. The site is plain static files, so it can be hosted for free (GitHub Pages) and serve any number of users. Photos never leave the user's device.

## 1. Get the trained weights

Train the model with `Bridge_Corrosion_Detection.ipynb` in Google Colab. It saves `/content/corrosion_unet_enhanced.pth`. Download it from Colab with:

```python
from google.colab import files
files.download("/content/corrosion_unet_enhanced.pth")
```

Put the file in `backend/`. It is ignored by git.

## 2. Convert the model for the browser

Use Python 3.11 or 3.12:

```powershell
cd E:\Bridge-Corrosion-Detection
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt onnx onnxruntime onnxscript
.\.venv\Scripts\python -m backend.export_onnx --weights backend\corrosion_unet_enhanced.pth
```

This writes `public/models/corrosion_unet.onnx` (about 31 MB). Batch normalisation is folded into the convolutions and the weights are stored as 8-bit integers, which is 4× smaller than the 124 MB full-precision model. The script compares the result with PyTorch and prints how many mask pixels agree. Add `--images path\to\valid\images` to compare on real photos. If agreement is below 99%, it warns you, and you can deploy the `--fp32` export instead.

## 3. Run locally

Requires Node.js 20.19+ or 22.12+.

```powershell
npm install
npm run dev
```

Open http://127.0.0.1:5173. The first analysis downloads the model once. Browsers keep it in their cache, so later visits start immediately.

- **WebGPU** is used when the browser and GPU support it (fastest, usually under a second per photo).
- Otherwise the model runs on the CPU with **WebAssembly** (several seconds per photo). If the GPU fails or hangs while starting, the page switches to the CPU on its own and remembers that for a week on that device.
- The CPU path uses several threads when the page is cross-origin isolated. The dev server sends the required headers. In production, `public/coi-sw.js` (a small service worker) adds them, which costs one automatic reload on the first visit.

## 4. Deploy for free on GitHub Pages

1. Commit `public/models/corrosion_unet.onnx` (31 MB is under GitHub's 100 MB file limit).
2. On GitHub, open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. Push to `main`. `.github/workflows/deploy.yml` runs the tests, builds the site and publishes it to `https://<user>.github.io/Bridge-Corrosion-Detection/`.

To keep the model out of the git repository, upload it to a free Hugging Face model repository instead. Then set a repository variable `MODEL_URL` (Settings → Secrets and variables → Actions → Variables) to its `.../resolve/main/corrosion_unet.onnx` URL.

## How the browser matches training

`src/inference/preprocess.js` is a JavaScript port of the training preprocessing: the notebook's unsharp mask (OpenCV's 5-tap Gaussian kernel with reflected borders) followed by `cv2.resize(..., (256, 256))` with OpenCV's fixed-point bilinear arithmetic. `tests/preprocess.test.js` checks it against values produced by OpenCV. A few values may differ by one brightness level because of float rounding.

Note: the notebook's last cell and the Python API resize with PIL/torchvision, which smooths differently from the OpenCV resize the model was trained with. The browser follows training.

## Python API (not used by the site)

`backend/app.py` is a standalone FastAPI service from earlier versions. The website no longer calls it; it runs everything in the browser. You can still start it for scripting or local experiments:

```powershell
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Notes

Coverage bands are UI heuristics: low <10%, moderate 10% to <25%, high ≥25%. They describe the fraction of image pixels classified as corrosion, not bridge structural severity or engineering safety. History is stored only in the browser (filenames, dates, source and coverage, up to 50 records). Photos and masks are not stored.

```powershell
npm test
npm run build
```
