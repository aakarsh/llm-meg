import epocher.llm_bert as B

def test_get_indexu_stimulus_stories():
    # Example usage:
    words = ["chicken"]
    rsa_matrix = B.get_index_stimulus_stories(words, 1)
    print("rsa_matrix shape", rsa_matrix.shape)
    assert rsa_matrix.shape == (len(words), 768)

