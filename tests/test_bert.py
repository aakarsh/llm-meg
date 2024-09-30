import epocher.llm_bert as B
import pytest 


def test_get_indexu_stimulus_stories():
    # Example usage:
    words = ["time", "mind", "fell"]
    _, word_embeddings = B.get_whole_word_embeddings(words, 1)
    assert word_embeddings.shape == (len(words), 768)
