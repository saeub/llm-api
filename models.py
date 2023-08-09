from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


Logprobs = list[dict[str, float]]


class Model:
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        stop_at: str | None = None,
        top_logprobs: int | None = None,
        logprobs_for_tokens: list[str] | None = None,
        **kwargs,
    ) -> tuple[str, Logprobs | None]:
        raise NotImplementedError()

    def classify(
        self,
        prompt: str,
        labels: list[str],
        **kwargs,
    ) -> dict[str, float]:
        raise NotImplementedError()


class ChatModel(Model):
    _end_of_response: str | None = None

    def chat(
        self,
        instructions: str,
        message: str,
        **kwargs,
    ) -> tuple[str, Logprobs | None]:
        prompt = self._build_prompt(instructions, message)
        if "stop_at" not in kwargs:
            kwargs["stop_at"] = self._end_of_response
        return self.generate(prompt, **kwargs)

    def _build_prompt(self, instructions: str, message: str) -> str:
        raise NotImplementedError()


class _StopPhraseCriteria(StoppingCriteria):
    def __init__(self, stop_phrase: str, tokenizer: PreTrainedTokenizerBase):
        self.stop_phrase = stop_phrase
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded.rstrip().endswith(self.stop_phrase.rstrip())


class _TransformersModel(Model):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):
        self.tokenizer = tokenizer
        self.model = model
        self._reverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = 100,
        stop_at: str | None = None,
        top_logprobs: int | None = None,
        logprobs_for_tokens: list[str] | None = None,
        **kwargs,
    ) -> tuple[str, Logprobs | None]:
        if max_tokens is None and stop_at is None:
            raise ValueError("Either max_tokens or stop_at must be specified")

        stopping_criteria = None
        if stop_at is not None:
            stopping_criteria = StoppingCriteriaList(
                [_StopPhraseCriteria(stop_phrase=stop_at, tokenizer=self.tokenizer)]
            )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            num_return_sequences=1,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        output_ids = output.sequences[0]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        output_text = output_text.removeprefix(prompt)

        if logprobs_for_tokens is not None:
            logprobs_for_token_ids = []
            for token in logprobs_for_tokens:
                token_ids = self.tokenizer.encode(token)
                if len(token_ids) > 1:
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                    raise ValueError(f"{token!r} is not a single token ({tokens!r}))")
                logprobs_for_token_ids.append(token_ids[0])
            logprobs_for_tokens = [self._reverse_vocab[token_id] for token_id in logprobs_for_token_ids]

        if top_logprobs is not None or logprobs_for_tokens is not None:
            output_logprobs = []
            for token_scores in output.scores:
                token_logprobs = token_scores.log_softmax(dim=-1)
                logprobs = {}
                if top_logprobs is not None:
                    top_token_logprobs, top_token_ids = token_logprobs.topk(top_logprobs, dim=-1)
                    logprobs.update({
                        self._reverse_vocab[token_id.item()]: score.item()
                        for token_id, score in zip(top_token_ids[0], top_token_logprobs[0])
                    })
                if logprobs_for_tokens is not None:
                    logprobs.update({
                        token: token_logprobs[0, token_id].item()
                        for token, token_id in zip(logprobs_for_tokens, logprobs_for_token_ids)
                    })
                output_logprobs.append(logprobs)

        else:
            output_logprobs = None

        return output_text, output_logprobs


class GPT2(_TransformersModel):
    def __init__(self):
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        super().__init__(tokenizer, model)


class Falcon(_TransformersModel):
    def __init__(self, size: Literal["7b", "40b"]):
        model_name = f"tiiuae/falcon-{size}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)


class FalconInstruct(_TransformersModel, ChatModel):
    _end_of_response = "\nUser:"

    def __init__(self, size: Literal["7b", "40b"]):
        model_name = f"tiiuae/falcon-{size}-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)

    def _build_prompt(self, instructions: str, message: str) -> str:
        # See https://huggingface.co/spaces/tiiuae/falcon-chat/blob/690a964f3c807fa1fc9acacef4f3d320fdbb6b0f/app.py#L41-L48
        return f"{instructions}\nUser: {message}\nAssistant:"


class Llama2(_TransformersModel):
    def __init__(self, size: Literal["7b", "13b", "70b"]):
        model_name = f"meta-llama/Llama-2-{size}-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)


class Llama2Chat(_TransformersModel, ChatModel):
    # See https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L44-L45
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    _end_of_response = B_INST

    def __init__(self, size: Literal["7b", "13b", "70b"]):
        model_name = f"meta-llama/Llama-2-{size}-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)

    def _build_prompt(self, instructions: str, message: str) -> str:
        # See https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L224-L269
        return f"{self.B_INST} {self.B_SYS}{instructions}{self.E_SYS}{message} {self.E_INST}"


def _get_subclasses(cls):
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_subclasses(subclass))
    return subclasses


MODEL_CLASSES: dict[str, type[Model]] = {
    subclass.__name__.lower(): subclass
    for subclass in _get_subclasses(Model)
    if not subclass.__name__.startswith("_") and subclass is not ChatModel
}
