import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from .env import *
import epocher.dataset as D
import epocher.rsa as rsa

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_average_rsa_from_correlations(correlations, noise_ceiling_map=None, word_pos=['VB']):
    """
    Plots the average RSA correlation for each task across subjects as a grouped bar plot for multiple models.

    :param correlations: List of dictionaries, each containing 'task_id', 'subject_id', 'model', 'correlation', and 'word_pos'.
    :param noise_ceiling_map: Dictionary containing 'task_id' as keys and tuples of (lower_bound, upper_bound) as values.
    :param word_pos: List of word parts of speech.
    """
    # Convert the list of correlations to a DataFrame for easier manipulation
    df = pd.DataFrame(correlations)

    # Set Seaborn's default style
    sns.set_theme(style="darkgrid")

    # Plot using Seaborn
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='task_id', y='correlation', hue='model', data=df, capsize=0.1,  errorbar='sd')

    # Add noise ceiling bounds if provided
    if noise_ceiling_map:
        for i, task_id in enumerate(df['task_id'].unique()):
            if task_id in noise_ceiling_map:
                lower_bound, upper_bound = noise_ceiling_map[task_id]
                # Position the noise ceiling to span across all models for a task
                ax.fill_between(
                    x=[i - 0.4, i + 0.4],
                    y1=lower_bound,
                    y2=upper_bound,
                    color='gray',
                    alpha=0.2,
                    label='Noise Ceiling' if i == 0 else ""
                )

    # Add labels and title
    plt.xlabel('Tasks')
    plt.ylabel('Average RSA Value')
    plt.title(f'Average RSA Values Across Tasks for Different Models (Word POS: {df["word_pos"].iloc[0]})')

    # Adjust legend to remove duplicate entries
    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))
    plt.legend(unique_handles_labels.values(), unique_handles_labels.keys(), title='Legend')
    average_rsa_plot_file = rsa.make_filename_prefix(f'average_rsa_plot.png', None, None, 
                        word_pos=word_pos, output_dir=IMAGES_DIR)

    plt.savefig(average_rsa_plot_file, bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_saved_similarity_matrix(subject_id=None, task_id=None, word_pos=None, sort_order=None):
    if not(subject_id == None and task_id == None):
        subject_id = [subject_id]
        task_id  = [task_id]
    elif subject_id == None and task_id == None:
        for subject_id in D.load_subject_ids():
            for task_id in D.load_task_ids():
                word_index, similarity_matrix = rsa.load_similarity_matrix(subject_id, task_id, word_pos=word_pos)
                sort_tag = ""
                if sort_order == "spectral": 
                    word_index, similarity_matrix = rsa.sort_by_spectral_clustering(word_index, similarity_matrix)
                    sort_tag = "_sort_spectral_"
                elif sort_order == "heirarchical":
                    word_index, similarity_matrix = rsa.sort_by_hierarchical_order(word_index, similarity_matrix)
                    sort_tag = "_sort_heirarchical_"
                else:
                    sort_tag=""
                similairty_matrix_image_path = rsa.make_filename_prefix(f'{sort_tag}similarity_matrix.png', subject_id, task_id, 
                        word_pos=word_pos, output_dir=IMAGES_DIR)
                plot_similarity_matrix(word_index, similarity_matrix, file_path=similairty_matrix_image_path)
    else:
        raise RuntimeError("Specify subject and task id, or leave both them empty")

def plot_similarity_matrix(word_index, similarity_matrix, 
        h=160, w=128, 
        cell_size=1.5, font_size_multiplier=0.4,
        file_path="./images/sim_words.png"):
    # Assuming 'similarity_matrix' is already computed
    # Set up the labels for the heatmap
    labels = word_index 

    # Calculate dynamic figure size based on number of words and the cell size
    num_words = len(word_index)
    fig_w = fig_h = num_words * cell_size
                    
    # Dynamically set the annotation font size based on cell size
    annot_fontsize = cell_size * font_size_multiplier * 10

    # Create the heatmap
    plt.figure(figsize=(h, w))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                        xticklabels=labels, yticklabels=labels, 
                        square=True, cbar_kws={"shrink": .8}, 
                        annot_kws={"size": annot_fontsize})

    # Title and labels
    plt.title("Cosine Similarity Matrix for Target Words")
    plt.xlabel("Target Words")
    plt.ylabel("Target Words")

    plt.savefig(file_path, bbox_inches='tight')
    # Show the plot
    plt.show()
    plt.close()

def plot_dendogram(word_index, similarity_matrix, h=80, w=64, file_path="./images/dendogram_words.png"):
    # Perform hierarchical clustering
    linkage_matrix = linkage(pdist(similarity_matrix), method='ward')

    # Create a dendrogram
    plt.figure(figsize=(80, 64))
    dendrogram(linkage_matrix, labels=word_index, leaf_rotation=90)
    plt.title('Dendrogram of Word Correlations')
    plt.savefig(file_path)
    plt.show()


def plot_tsne(word_index, similarity_matrix):
    # Assuming 'similarity_matrix' is already computed and 'target_words' contains word labels

    # Step 1: Compute the t-SNE Representation in 3D
    tsne_model = TSNE(n_components=3, random_state=42, perplexity=5, n_iter=1000, metric="cosine")
    tsne_results = tsne_model.fit_transform(similarity_matrix)


    # Create a DataFrame with t-SNE results and words
    tsne_df = pd.DataFrame(tsne_results, columns=['Component 1', 'Component 2', 'Component 3'])
    tsne_df['Words'] = word_index 

    # Step 3: Create the 3D scatter plot using Plotly
    fig = px.scatter_3d(tsne_df,
                         x='Component 1',
                         y='Component 2',
                         z='Component 3',
                         text='Words',
                         title='3D t-SNE Visualization of Target Words',
                         labels={'Component 1': 't-SNE Component 1',
                                 'Component 2': 't-SNE Component 2',
                                 'Component 3': 't-SNE Component 3'},
                         template='plotly_white')

    # Step 4: Customize the layout
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(scene=dict(
                        xaxis_title='t-SNE Component 1',
                        yaxis_title='t-SNE Component 2',
                        zaxis_title='t-SNE Component 3'),
                      margin=dict(l=0, r=0, b=0, t=40))

    # Step 5: Show the plot
    fig.show()
