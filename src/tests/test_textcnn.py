import pytest


torch = pytest.importorskip("torch")

from news_sentinel.models.textcnn import TextCNN
from news_sentinel.models.torch_text_data import build_vocabulary


def test_build_vocab_contains_special_tokens() -> None:
    vocab = build_vocabulary(["hello world", "hello sports"]) 
    assert vocab.token_to_idx["<pad>"] == 0
    assert vocab.token_to_idx["<unk>"] == 1


def test_textcnn_forward_shape() -> None:
    model = TextCNN(vocab_size=50, embedding_dim=16, num_filters=8, num_classes=4)
    batch = torch.randint(low=0, high=50, size=(5, 20))
    logits = model(batch)
    assert logits.shape == (5, 4)
