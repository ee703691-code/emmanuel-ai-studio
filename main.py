import os
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI()

API_KEY = os.environ.get("WAVESPEED_API_KEY", "")
BASE = "https://api.wavespeed.ai/api/v3"
SIZES = {"Square": "1024*1024", "Wide": "1280*720", "Tall": "720*1280"}


class ImageReq(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    shape: str = "Square"


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/api/image")
def make_image(req: ImageReq):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="The server is missing its API key.")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "prompt": req.text,
        "size": SIZES.get(req.shape, "1024*1024"),
        "output_format": "jpeg",
    }

    try:
        r = requests.post(
            f"{BASE}/wavespeed-ai/qwen-image/text-to-image",
            json=body, headers=headers, timeout=60,
        )
        r.raise_for_status()
        task_id = r.json()["data"]["id"]

        for _ in range(60):
            time.sleep(2)
            res = requests.get(
                f"{BASE}/predictions/{task_id}/result", headers=headers, timeout=30
            )
            res.raise_for_status()
            data = res.json()["data"]
            if data["status"] == "completed":
                return {"url": data["outputs"][0]}
            if data["status"] == "failed":
                raise HTTPException(status_code=502, detail=data.get("error") or "Generation failed.")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail="Could not reach the image service: " + str(e))
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected reply from the image service.")

    raise HTTPException(status_code=504, detail="The image took too long. Please try again.")
