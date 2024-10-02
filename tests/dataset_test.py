import epocher.dataset as D


def test_load_subject_information():
    assert len(D.load_subject_information())  > 0

def test_get_epoch_word_map():
    word_index, words_sorted_metadata_df, target_word_epochs = D._get_epoch_word_map('01', 0, 0)
    assert len(word_index) == len(target_word_epochs.keys())
    
