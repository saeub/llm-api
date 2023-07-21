import argparse
import os
from ast import literal_eval
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import models

def kwarg_eval(string: str) -> tuple[str, Any]:
    key, value = string.split("=")
    return (key, literal_eval(value))

parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=models.MODEL_CLASSES.keys())
parser.add_argument("--kwargs", type=kwarg_eval, nargs="*", default=[])
parser.add_argument("--gpus", type=int, nargs="*", default=[])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

model = models.MODEL_CLASSES[args.model_name](**dict(args.kwargs))

app = FastAPI()


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int | None = None
    stop_at: str | None = None
    config: dict[str, Any] = {}


@app.post("/generate")
def generate(request: GenerationRequest):
    return model.generate(
        request.prompt, max_tokens=request.max_tokens, stop_at=request.stop_at
    )


class ClassificationRequest(BaseModel):
    prompt: str
    labels: list[str]
    config: dict[str, Any] = {}


@app.post("/classify")
def classify(request: ClassificationRequest):
    return model.classify(request.prompt, request.labels)


uvicorn.run(app)
