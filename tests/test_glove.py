import epocher.llm_glove as G

def test_llm_glove():
    # Example usage:
    words = ['cat', 'dog', 'apple', 'banana']
    rsa_matrix = G.create_rsa_matrix(words)
    assert rsa_matrx.shape == (len(words), len(words))

