import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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

    plt.savefig('./images/cov.png')
    # Show the plot
    plt.show()i



