import epocher.dataset as D

def test_get_target_word_vectors():
    word_index, target_word_vectors = D._get_target_word_vectors(subject_id='01', session_id=0, task_id=0)
    print(target_word_vectors.shape)
    assert len(target_word_vectors) > 0
    assert (72, 9568) == target_word_vectors.shape

def test_get_target_word_vectors_direct():
    word_index, target_word_vectors = D._get_target_word_vectors(subject_id='01', session_id=0, task_id=0, use_ica=False)
    print(target_word_vectors.shape)
    assert len(target_word_vectors) > 0
    assert (72, 9568) == target_word_vectors.shape


def test_load_subject_information():
    assert len(D.load_subject_information())  > 0

def test_get_epoch_word_map():
    word_index, words_sorted_metadata_df, target_word_epochs = D._get_epoch_word_map('01', 0, 0)
    assert len(word_index) == len(target_word_epochs.keys())

def test_segment_word_epcoh_map():
    word_index, words_sorted_metadata_df, target_word_epochs = D._get_epoch_word_map('02', 0, 0)
    n_segments=10
    segmented_epochs = D._segment_word_epoch_map(n_segments, word_index,  target_word_epochs)
    assert len(segmented_epochs[word_index[0]]) == n_segments


def test_segment_ica_epochmap():
    subject_id='03'
    session_id=0
    task_id=0
    n_components=15
    tmax=0.25
    word_index, word_metadata_df, word_epoch_map, ica_epochs = \
          D._get_ica_epochs(subject_id, session_id, task_id,n_components=n_components, tmax=tmax)

    n_segments=10
    segmented_epochs = D._segment_word_epoch_map(n_segments, word_index, ica_epochs )
    assert len(segmented_epochs[word_index[0]]) == n_segments


