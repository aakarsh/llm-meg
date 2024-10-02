import epocher.stories as S
import epocher.dataset as D

def test_always_pass():
    pass

def test_load_raw_meta():
    raw_file = D._get_raw_file('01', 0, 0)
    meta_data = D._load_raw_meta(raw_file)
    m = meta_data.query('kind == "word"')
    assert len(m) > 0
 
def test_construct_subject_rsa():
    raw_file = D._get_raw_file('01', 0, 0)
    meta_data = D._load_raw_meta(raw_file)
    word_epochs = D.segment_by_word(raw_file) 
    assert len(word_epochs) > 0
    assert len(D._word_epoch_words(word_epochs.metadata)) > 0

def test_get_segmented_similarity_matrix():
    retval = R._get_segmented_similarity_matrix(subject_id='01', session_id=0, task_id=0, 
                                        n_segments=10, n_components=15, tmax=0.25, 
                                        reference_word_idx = None, save_similarity_matrix=False, 
                                        debug=False):
    assert retval

