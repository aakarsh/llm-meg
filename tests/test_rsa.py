import epocher.stories as S
import epocher.dataset as D
import epocher.rsa as R


def test_noise_celing_per_story():
    word_index, avg_rdm = R.compute_average_rdm_for_task_id(0)
    assert avg_rdm.shape == (len(word_index), len(word_index))
    low, upper = R.compute_noise_ceiling_bounds(0)
    assert low > 0 and upper > 0
    print("low", low, "upper", upper)

def test_per_electrode_rsa():
    rsa_matrices_per_electrode, time_points = R.sliding_window_rsa_per_electrode()
    print("time_points", len(time_points))
    print("rsa_matrices_per_electrode:", len(time_points))
    print("rsa_matrices_per_electrode.keys():", rsa_matrices_per_electrode.keys())
    first_key = list(rsa_matrices_per_electrode.keys())[0]
    print(rsa_matrices_per_electrode[first_key].shape)

def test_plot_rsa_topo_over_time():
    R.plot_rsa_topomap_over_time('01', 0)

def test_plot_rsa_linepolot_over_time():
    R.plot_rsa_lineplot_over_time('01', 0)


def test_plot_rsa_lineplot_per_channel():
    R.plot_rsa_lineplot_per_channel('01', 0)

def test_compute_model_p_value():
    for task_id in [0, 1,  3]:
        rsa_score_b, p_value_b = R.compute_model_p_value(task_id, model="BERT")
        rsa_score_g, p_value_g = R.compute_model_p_value(task_id, model="GLOVE")
        print(f"BERT {task_id}: {rsa_score_b}, pvalue: {p_value_b}")
        print(f"GLOVE {task_id}: {rsa_score_g}, pvalue: {p_value_g}")

def test_sliding_window_rsa_per_electrode():
    print("test_sliding_window_rsa_per_electrode")
    results, times = R.sliding_window_rsa_per_electrode('01', task_id=0)
    pass

def test_get_similarity_matrix_nouns():
    """
    """
    word_index, similarity_matrices = R._get_similarity_matrix(subject_id='01', session_id=0, task_id=0,
                                        n_components=15, tmax=0.25,
                                        reference_word_idx = None,
                                        save_similarity_matrix=False,
                                        word_pos=['NN'],
                                        debug=False)
    assert len(similarity_matrices) > 0
    assert (len(word_index), len(word_index)) == similarity_matrices.shape


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
    word_index, similarity_matrices = R._get_segmented_similarity_matrix(subject_id='01', session_id=0, task_id=0, 
                                        n_segments=10, n_components=15, tmax=0.25, 
                                        reference_word_idx = None, save_similarity_matrix=False, 
                                        debug=False)
    assert len(similarity_matrices) > 0
    assert (10, len(word_index), len(word_index)) == similarity_matrices.shape
 


def test_plot_similarity_matrix():
    pass
