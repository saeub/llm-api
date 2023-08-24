from typing import NamedTuple
import pytest

from models import Message


class MockTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def get_vocab(self, *args, **kwargs):
        return {}

    def encode(self, *args, **kwargs):
        return [0, 1, 2]

    def decode(self, *args, **kwargs):
        return "decoded"


class MockModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        Output = NamedTuple("Output", [("sequences", str), ("logprobs", None)])
        return Output("generated", None)


@pytest.fixture
def mock_transformers(monkeypatch):
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return MockTokenizer()

    def mock_model_from_pretrained(*args, **kwargs):
        return MockModel()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", mock_tokenizer_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_model_from_pretrained
    )


def test_falcon(mock_transformers):
    from models import Falcon

    model = Falcon("70b")
    assert model.generate("prompt", max_tokens=10) == ("decoded", None)


def test_falcon_instruct(mock_transformers):
    from models import FalconInstruct

    model = FalconInstruct("70b")
    messages = [Message.user("message")]
    assert model._build_prompt(None, messages) == "User: message\nAssistant:"
    assert (
        model._build_prompt("instructions", messages)
        == "instructions\nUser: message\nAssistant:"
    )
    messages.extend([Message.assistant("response"), Message.user("another message")])
    assert (
        model._build_prompt(None, messages)
        == "User: message\nAssistant: response\nUser: another message\nAssistant:"
    )
    assert (
        model._build_prompt("instructions", messages)
        == "instructions\nUser: message\nAssistant: response\nUser: another message\nAssistant:"
    )
    assert model.chat(messages) == ("decoded", None)
    assert model.chat(messages, instructions="instructions") == ("decoded", None)
    assert model.generate("prompt", max_tokens=10) == ("decoded", None)


def test_llama2(mock_transformers):
    from models import Llama2

    model = Llama2("70b")
    assert model.generate("prompt", max_tokens=10) == ("decoded", None)


def test_llama2_chat(mock_transformers):
    from models import Llama2Chat

    model = Llama2Chat("70b")
    messages = [Message.user("message")]
    assert model._build_prompt(None, messages) == "[INST] message [/INST]"
    assert (
        model._build_prompt("instructions", messages)
        == "[INST] <<SYS>>\ninstructions\n<</SYS>>\n\nmessage [/INST]"
    )
    messages.extend([Message.assistant("response"), Message.user("another message")])
    assert (
        model._build_prompt(None, messages)
        == "[INST] message [/INST] response</s><s>[INST] another message [/INST]"
    )
    assert (
        model._build_prompt("instructions", messages)
        == "[INST] <<SYS>>\ninstructions\n<</SYS>>\n\nmessage [/INST] response</s><s>[INST] another message [/INST]"
    )
    assert model.chat(messages) == ("decoded", None)
    assert model.chat(messages, instructions="instructions") == ("decoded", None)
    assert model.generate("prompt", max_tokens=10) == ("decoded", None)
