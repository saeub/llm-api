# LLM API

This is a very simple web API to run various LLMs for inference. Bugs included.

## Models

| Model name     | Sizes                 |
| -------------- | --------------------- |
| llama2         | 7b, 13b, 70b          |
| llama2chat     | 7b, 13b, 70b          |
| falcon         | 7b, 40b               |
| falconinstruct | 7b, 40b               |
| gpt2           | -- (just for testing) |

## Example usage

```bash
python api.py llama2chat --kwargs size=70b --gpus 0 1 2 3
```
