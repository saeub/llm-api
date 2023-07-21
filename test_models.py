import pytest


class MockTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        return [0, 1, 2]

    def decode(self, *args, **kwargs):
        return "decoded"


class MockModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return "generated"


@pytest.fixture
def mock_transformers(monkeypatch):
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return MockTokenizer()
    
    def mock_model_from_pretrained(*args, **kwargs):
        return MockModel()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_tokenizer_from_pretrained)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_model_from_pretrained)


def test_falcon(mock_transformers):
    from models import Falcon

    model = Falcon("70b")
    assert model.generate("prompt", max_tokens=10) == "decoded"


def test_falcon_instruct(mock_transformers):
    from models import FalconInstruct

    model = FalconInstruct("70b")
    assert model._build_prompt("instructions", "message") == "instructions\nUser: message\nAssistant:"
    assert model.respond("instructions", "message", max_tokens=10) == "decoded"
    assert model.generate("prompt", max_tokens=10) == "decoded"


def test_llama2(mock_transformers):
    from models import Llama2

    model = Llama2("70b")
    assert model.generate("prompt", max_tokens=10) == "decoded"


def test_llama2_chat(mock_transformers):
    from models import Llama2Chat

    model = Llama2Chat("70b")
    assert model._build_prompt("instruction", "message") == "[INST] <<SYS>>\ninstruction\n<</SYS>>\n\nmessage [/INST]"
    assert model.respond("instruction", "message", max_tokens=10) == "decoded"
    assert model.generate("prompt", max_tokens=10) == "decoded"
