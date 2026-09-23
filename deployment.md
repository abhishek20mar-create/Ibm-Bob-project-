# IBM Cloud Deployment Guide

This document covers deploying the Heart Attack Prediction API to two IBM Cloud compute targets:

| Option | Best For |
|---|---|
| **IBM Cloud Code Engine** | Serverless containers; pay-per-use; no cluster management |
| **IBM Cloud Foundry** | PaaS push-to-deploy; managed runtime |

---

## Prerequisites

```bash
# 1. IBM Cloud CLI
curl -fsSL https://clis.cloud.ibm.com/install/linux | sh

# 2. Code Engine plugin (for Option A)
ibmcloud plugin install code-engine

# 3. Container Registry plugin (for Option A)
ibmcloud plugin install container-registry

# 4. Login
ibmcloud login --sso
ibmcloud target -r us-south -g Default
```

---

## Option A — IBM Cloud Code Engine (Recommended)

Code Engine runs your container image serverlessly with automatic scaling and built-in HTTPS.

### Step 1 — Build & push container image

```bash
cd heart_attack_prediction

# Tag for IBM Container Registry
export ICR_NS=heart-prediction          # your ICR namespace
export IMAGE=us.icr.io/$ICR_NS/heart-attack-api:latest

# Create namespace (once)
ibmcloud cr namespace-add $ICR_NS

# Build & push
ibmcloud cr login
docker build -t $IMAGE .
docker push $IMAGE
```

### Step 2 — Create a Code Engine project

```bash
ibmcloud ce project create --name heart-prediction-project
ibmcloud ce project select  --name heart-prediction-project
```

### Step 3 — Deploy the application

```bash
ibmcloud ce application create \
  --name heart-attack-api \
  --image $IMAGE \
  --registry-secret icr-secret \
  --port 8000 \
  --min-scale 0 \
  --max-scale 3 \
  --cpu 0.5 \
  --memory 1G \
  --env PORT=8000
```

### Step 4 — Get the public URL

```bash
ibmcloud ce application get --name heart-attack-api --output url
# → https://heart-attack-api.<hash>.us-south.codeengine.appdomain.cloud
```

### Step 5 — Verify

```bash
curl https://<your-url>/
# {"status":"ok","message":"Heart Attack Prediction API is running"}

curl -X POST https://<your-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,
       "restecg":0,"thalach":160,"exang":0,"oldpeak":1.4,
       "slope":2,"ca":0,"thal":3}'
```

### Step 6 — Update deployment (redeploy on new image)

```bash
docker build -t $IMAGE . && docker push $IMAGE
ibmcloud ce application update --name heart-attack-api --image $IMAGE
```

---

## Option B — IBM Cloud Foundry

Cloud Foundry handles the runtime stack automatically from your source code — no Docker build step needed.

### Step 1 — Target a CF space

```bash
ibmcloud target --cf
ibmcloud cf target -o <your-org> -s <your-space>
```

### Step 2 — Train the model locally first

The `manifest.yml` runs `python train.py` at startup, but the first boot will be slow.  
For faster startup, train locally and commit the `model/` artefacts:

```bash
cd heart_attack_prediction
pip install -r requirements.txt
python train.py
# model/pipeline.pkl, stats.json, feature_metadata.json are now present
```

### Step 3 — Push

```bash
ibmcloud cf push
```

Cloud Foundry reads `manifest.yml` automatically. The app will be available at:
```
https://heart-attack-prediction-api.<region>.cf.appdomain.cloud
```

### Step 4 — Check logs

```bash
ibmcloud cf logs heart-attack-prediction-api --recent
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the API listens on (set automatically by Code Engine / CF) |
| `PYTHONPATH` | `.` | Ensures `src/` is importable |

---

## Scaling

### Code Engine

```bash
# Scale to 0 (serverless) when idle, max 5 replicas under load
ibmcloud ce application update \
  --name heart-attack-api \
  --min-scale 0 \
  --max-scale 5
```

### Cloud Foundry

```bash
ibmcloud cf scale heart-attack-prediction-api -i 3   # 3 instances
```

---

## IBM Watson AutoAI Integration (Optional)

To replace the scikit-learn pipeline with an IBM Watson AutoAI-trained model:

1. Upload `data/heart.csv` to an IBM Cloud Object Storage bucket.
2. Open **Watson Studio** → **New AutoAI Experiment**.
3. Select `num` as the prediction column, binary classification.
4. Run the experiment; AutoAI selects the best pipeline automatically.
5. Download the generated model as a **pickle** or **PMML** file.
6. Replace `model/pipeline.pkl` with the AutoAI export.
7. Ensure the exported pipeline exposes `.predict()` and `.predict_proba()` — if not, wrap it in a thin sklearn-compatible adapter.

AutoAI REST endpoint (alternative to local model):

```python
# In app.py, replace local pipeline inference with IBM Watson ML scoring:
import requests as req

WML_URL   = "https://us-south.ml.cloud.ibm.com"
SPACE_ID  = "<your-deployment-space-id>"
DEPLOY_ID = "<your-autoai-deployment-id>"
IAM_TOKEN = "<bearer-token>"   # ibmcloud iam oauth-tokens

def predict_via_wml(features: list) -> dict:
    payload = {"input_data": [{"fields": FEATURE_COLS, "values": [features]}]}
    headers = {"Authorization": f"Bearer {IAM_TOKEN}",
               "Content-Type": "application/json"}
    url = f"{WML_URL}/ml/v4/deployments/{DEPLOY_ID}/predictions?version=2021-05-01"
    resp = req.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["predictions"][0]
```
