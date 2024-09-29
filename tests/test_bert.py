import epocher.llm_bert as B

def test_llm_glove():
    # Example usage:
    #words = ['cat', 'dog', 'apple', 'banana']
    rsa_matrix = B.get_index_stimulus_stories(None)
    print("rsa_matrix shape", rsa_matrix.shape)
    #assert rsa_matrix.shape == (len(words), len(words))

