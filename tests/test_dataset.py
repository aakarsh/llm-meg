import epocher.dataset as D


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
    word_index, words_sorted_metadata_df, target_word_epochs, ica_epochs = \
      word_index, word_metadata_df, word_epoch_map, ica_epochs = \
          D._get_ica_epochs(subject_id, session_id, task_id,n_components=n_components, tmax=tmax)

    n_segments=10
    segmented_epochs = D._segment_word_epoch_map(n_segments, word_index, ica_epochs )
    assert len(segmented_epochs[word_index[0]]) == n_segments


