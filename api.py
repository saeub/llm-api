import argparse
from ast import literal_eval
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import models

parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=models.MODEL_CLASSES.keys())
parser.add_argument("--kwargs", type=literal_eval, default={})
args = parser.parse_args()


model = models.MODEL_CLASSES[args.model_name](**args.kwargs)

app = FastAPI()


class GenerationRequest(BaseModel):
    text: str
    max_tokens: int | None = None
    stop_at: str | None = None
    config: dict[str, Any]


@app.post("/generate")
def generate(request: GenerationRequest):
    return model.generate(
        request.text, max_tokens=request.max_tokens, stop_at=request.stop_at
    )


class ClassificationRequest(BaseModel):
    text: str
    labels: list[str]
    config: dict[str, Any]


@app.post("/classify")
def classify(request: ClassificationRequest):
    return model.classify(request.text, request.labels)


uvicorn.run(app)
