import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

import pandas as pd
import json
import os

from .env import *
import epocher.dataset as D

def word_index_file_name(subject_id, task_id):
   return f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_word_index.json'

def similarity_index_file_name(subject_id, task_id):
    return f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'

def load_similarity_matrix(subject_id, task_id):
    word_index_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_word_index.json'
    similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'

    word_index = None
    with open(word_index_file, 'r') as infile:
         word_index = json.load(infile)

    similarity_matrix = np.load(similarity_matrix_file)

    return word_index, similarity_matrix
    

def plot_saved_similarity_matrix(subject_id=None, task_id=None):

    if subject_id == None and task_id == None:
        for subject_id in D.load_subject_ids():
            for task_id in D.load_task_ids():
                if os.path.exists(word_index_file_name(subject_id, task_id)) and \
                    os.path.exists(similarity_index_file_name(subject_id, task_id)):

                    file_path = f"./images/subject_id_{subject_id}_task_id_{task_id}_similarity_matrix.png" 
                    word_index, similarity_matrix = load_similarity_matrix(subject_id, task_id)
                    plot_similarity_matrix(word_index, similarity_matrix, file_path=file_path)
    else: # single 
        if os.path.exists(word_index_file_name(subject_id, task_id)) and \
            os.path.exists(similarity_index_file_name(subject_id, task_id)):
            word_index, similarity_matrix = load_similarity_matrix(subject_id, task_id)
            file_path = f"./images/subject_id_{subject_id}_task_id_{task_id}_similarity_matrix.png" 
            plot_similarity_matrix(word_index, similarity_matrix, file_path=file_path)


def plot_similarity_matrix(word_index, similarity_matrix, h=160, w=128, file_path="./images/sim_words.png"):
    # Assuming 'similarity_matrix' is already computed
    # Set up the labels for the heatmap
    labels = word_index #[word_epochs[i].metadata['word'].values[0] for i in target_words_index]

    # Create the heatmap
    plt.figure(figsize=(h, w))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                        xticklabels=labels, yticklabels=labels, square=True, cbar_kws={"shrink": .8})

    # Title and labels
    plt.title("Cosine Similarity Matrix for Target Words")
    plt.xlabel("Target Words")
    plt.ylabel("Target Words")

    plt.savefig(file_path)
    # Show the plot
    plt.show()


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
