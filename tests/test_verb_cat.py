import epocher.verb_cat as VC


def test_something():
    verb_features = VC.mark_verb_features(2 ,["violating"])
    print(verb_features)
    assert len(verb_features) > 0
