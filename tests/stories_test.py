import epocher.stories as S

def test_always_pass():
    # when you lose all hope always pass is there for you.
    assert True == True
    assert True != False
    pass

def test_stories():
    assert len(S.load_experiment_stories().keys() ) == 4

def test_salient_words():
    assert len(S.load_experiment_salient_words()) == 10
