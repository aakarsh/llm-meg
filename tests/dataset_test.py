import epocher.dataset as D

def test_always_pass():
    pass

def test_load_subject_information():
    assert len(D.load_subject_information())  > 0
