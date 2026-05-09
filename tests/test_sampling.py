from fin_graph_rag.utils.text import stable_hash, tokenize


def test_stable_hash_repeatable():
    assert stable_hash("a", 1) == stable_hash("a", 1)


def test_tokenize_basic():
    assert tokenize("Gross margin increased 10%") == ["gross", "margin", "increased", "10%"]
