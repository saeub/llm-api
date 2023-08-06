import argparse
import os
from ast import literal_eval
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import models


def kwarg_eval(string: str) -> tuple[str, Any]:
    key, value = string.split("=")
    try:
        value = literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    return (key, value)


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
        request.prompt,
        max_tokens=request.max_tokens,
        stop_at=request.stop_at,
        **request.config,
    )


class ChatRequest(BaseModel):
    instructions: str
    message: str
    max_tokens: int | None = None
    stop_at: str | None = None
    config: dict[str, Any] = {}


@app.post("/chat")
def chat(request: ChatRequest):
    if not isinstance(model, models.ChatModel):
        raise HTTPException(
            status_code=400, detail=f"{model.__class__.__name__} is not a chat model"
        )
    return model.chat(
        request.instructions,
        request.message,
        max_tokens=request.max_tokens,
        stop_at=request.stop_at,
        **request.config,
    )


class ClassificationRequest(BaseModel):
    prompt: str
    labels: list[str]
    config: dict[str, Any] = {}


@app.post("/classify")
def classify(request: ClassificationRequest):
    return model.classify(request.prompt, request.labels)


uvicorn.run(app)
