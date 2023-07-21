from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Model:
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        stop_at: str | None = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError()

    def classify(
        self,
        prompt: str,
        labels: list[str],
        **kwargs,
    ) -> dict[str, float]:
        raise NotImplementedError()


class _StopPhraseCriteria(StoppingCriteria):
    def __init__(self, stop_phrase: str, tokenizer: PreTrainedTokenizerBase):
        self.stop_phrase = stop_phrase
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded.endswith(self.stop_phrase)


class _TransformersModel(Model):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):
        self.tokenizer = tokenizer
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        stop_at: str | None = None,
        **kwargs,
    ) -> str:
        if max_tokens is None and stop_at is None:
            raise ValueError("Either max_tokens or stop_at must be specified")

        stopping_criteria = None
        if stop_at is not None:
            stopping_criteria = StoppingCriteriaList(
                [_StopPhraseCriteria(stop_phrase=stop_at, tokenizer=self.tokenizer)]
            )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            input_ids,
            num_return_sequences=1,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
    def __init__(self, size: Literal["7b", "40b"], instruct: bool = False):
        model_name = f"tiiuae/falcon-{size}"
        if instruct:
            model_name += "-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)


class Llama2(_TransformersModel):
    def __init__(self, size: Literal["7b", "13b", "70b"], chat: bool = False):
        model_name = f"meta-llama/Llama-2-{size}-hf"
        if chat:
            model_name += "-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        super().__init__(tokenizer, model)


def _get_subclasses(cls):
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_subclasses(subclass))
    return subclasses


MODEL_CLASSES = {
    subclass.__name__.lower(): subclass
    for subclass in _get_subclasses(Model)
    if not subclass.__name__.startswith("_")
}
