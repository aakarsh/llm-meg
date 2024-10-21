#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import epocher
import epocher.rsa as rsa
import epocher.dataset as D
import epocher.plots as P
from epocher.env import *

def compute_rsa_similarity(subject1, subject2, task_id):
    """
    Compute the RSA similarity between two subjects for a given task.
    
    :param subject1: ID of the first subject
    :param subject2: ID of the second subject
    :param task_id: ID of the task to compare
    :return: Correlation between RSA matrices of the two subjects
    """
    correlation = rsa._compare_subjects(subject1, subject2, task_id=task_id)
    return correlation

def compare_with_model(model, save_comparisons=True, word_pos=['VB']):
    subject_ids = D.load_subject_ids()
    task_ids = D.load_task_ids() 
    correlation_comparisons = np.zeros((len(subject_ids), len(task_ids)))
    for subject_id in subject_ids:
        for task_id in task_ids: 
             correlation = rsa._compare_with_model(subject_id, task_id, model=model, word_pos=word_pos)
             print(f"Comparing {subject_id}, {task_id}: {correlation} with {model}, {word_pos}")
             correlation_comparisons[(int(subject_id)-1, int(task_id)-1)] = correlation 

    # summarize this as a plot ?
    comparison_file_name = f'{OUTPUT_DIR}/model_comparison_{model}_similarity_matrix.npy'
    np.save(comparison_file_name, correlation_comparisons)

    return correlation_comparisons

def compare_with_model_layers_segmented(model, save_comparisons=True):
    subject_ids = D.load_subject_ids()
    task_ids = D.load_task_ids() 
    correlation_comparisons = np.zeros((len(subject_ids), len(task_ids)))
    for subject_id in subject_ids:
        for task_id in task_ids: 
             correlation = rsa._compare_segments_with_model_layers(subject_id, task_id, model=model)
             print(f"Comparing {subject_id}, {task_id}: {correlation} with {model}", correlation)
             # correlation_comparisons[(int(subject_id)-1, int(task_id)-1)] = correlation 
    # comparison_file_name = f'{OUTPUT_DIR}/model_comparison_{model}_similarity_matrix.npy'
    # np.save(comparison_file_name, correlation_comparisons)


def compute_all_rsa_matrics(task_id=None, segmented=False, word_pos =["VB"]):
    subject_ids = D.load_subject_ids()
    task_ids = D.load_task_ids() if not task_id else [task_id] 

    for subject_id in subject_ids:
        for task_id in task_ids: 
            if not segmented:
                rsa._get_similarity_matrix(subject_id=subject_id, task_id=task_id, 
                        word_pos=word_pos, save_similarity_matrix=True)
            else:
                print("Generating segmented matrices")
                rsa._get_segmented_similarity_matrix(subject_id=subject_id, 
                                                        task_id=task_id, 
                                                        save_similarity_matrix=True)
  

def compute_similarity_matrics(task_id=None, model='GLOVE', hidden_layer=None, word_pos=['VB']):
    subject_ids = D.load_subject_ids()
    task_ids = D.load_task_ids() if not task_id else [task_id] 

    for subject_id in subject_ids:
        for task_id in task_ids: 
             similarity_matrix_0 = \
                rsa.compute_similarity_matrics(subject_id, task_id, 
                        model=model, save_similarity_matrix=True, 
                        hidden_layer=hidden_layer, 
                        word_pos = word_pos)


def compute_rsa_matrix(subject_id, task_id):
    word_index, similarity_matrix_0 = \
        rsa._get_similarity_matrix(subject_id=subject_id, task_id=task_id, save_similarity_matrix=True)
    return word_index, similarity_matrix_0


def main():
    # Parse arguments
    # Initialize the main parser
    parser = argparse.ArgumentParser(description='Perform RSA analysis.')
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')

    # Subcommand for comparing two subjects
    compare_parser = subparsers.add_parser('compare', help='Compare RSA similarity between two subjects')
    compare_parser.add_argument('--subject1', type=str, required=True, help='ID of the first subject')
    compare_parser.add_argument('--subject2', type=str, required=True, help='ID of the second subject')
    compare_parser.add_argument('--task-id', type=int, required=True, help='ID of the task to compare')

    # Subcommand for generating similarity matrix for one subject
    generate_parser = subparsers.add_parser('generate', help='Generate similarity matrix for a subject')
    generate_parser.add_argument('--subject-id', type=str, required=True, help='ID of the subject')
    generate_parser.add_argument('--task-id', type=int, required=True, help='ID of the task')
    generate_parser.add_argument('--word-pos', default='VB', type=str, required=False, help='Filter words by part of speech')

    generate_all_parser = subparsers.add_parser('generate-all', help='Generate all similarity matrics for all tasks.')
    generate_all_parser.add_argument('--segmented', type=bool, required=False, help='Segment word into parts, each with its won similairty matrix', default=None)
    generate_all_parser.add_argument('--word-pos',default='VB', type=str, required=False, help='Filter words by part of speech')
    
    plot_rsa_tabe_parser = subparsers.add_parser('plot-rsa-table', help='Plot RSA Confusion Table for a sobject and task, use cached results')
    plot_rsa_tabe_parser.add_argument('--subject-id', type=str, required=False, help='ID of the subject', default=None)
    plot_rsa_tabe_parser.add_argument('--task-id', type=int, required=False, help='ID of the task', default=None)
    plot_rsa_tabe_parser.add_argument('--sort-order', type=str, required=False, help='ID of the task', default=None)
    plot_rsa_tabe_parser.add_argument('--word-pos',default='VB', type=str, required=True, help='Filter words by part of speech')

    generate_model = subparsers.add_parser('generate-model', help='Generate all similarity matrics for all tasks.')
    generate_model.add_argument('--model', type=str, required=True, help='Model Name', default=None)
    generate_model.add_argument('--hidden-layer', type=int, required=False, help='Model Name', default=None)
    generate_model.add_argument('--word-pos',default=None, type=str, required=False, help='Filter words by part of speech')

    compare_model = subparsers.add_parser('compare-model', help='Generate comparisons')
    compare_model.add_argument('--model', type=str, required=True, help='Model Name', default=None)
    compare_model.add_argument('--word-pos',default='VB', type=str, required=True, help='Filter words by part of speech')
    
    compare_segmented_model_layers = subparsers.add_parser('compare-segmented-model-layers', help='Generate comparisons')
    compare_segmented_model_layers.add_argument('--model', type=str, required=True, help='Model Name', default=None)

    average_rsa_bar_plots = subparsers.add_parser('plot-bar-average-rsa', help='regenerate rsa plots')
    average_rsa_bar_plots.add_argument('--word-pos',default='VB', type=str, required=True, help='Filter words by part of speech')
 
    args = parser.parse_args()


    if args.command == 'compare':
        _compare_subjects(args.subject1, args.subject2, task_id=args.task_id)
        correlation = compute_rsa_similarity(args.subject1, args.subject2, args.task_id)
        # Output the result
        print(f"RSA correlation between subject {args.subject1} and {args.subject2} for task {args.task_id}: {correlation:.2f}")
    elif args.command == 'generate':
        compute_rsa_matrix(subject_id=args.subject_id, task_id=args.task_id)
    elif args.command == 'generate-all':
        compute_all_rsa_matrics(segmented=args.segmented, word_pos=args.word_pos.split(","))
    elif args.command == 'generate-model':
       compute_similarity_matrics(model=args.model, hidden_layer=args.hidden_layer, word_pos=args.word_pos.split(","))
    elif args.command == 'compare-model':
       compare_with_model(args.model, word_pos=args.word_pos.split(","))
    elif args.command == 'compare-segmented-model-layers':
        compare_with_model_layers_segmented(args.model, word_pos=args.word_pos.split(","))
    elif args.command == 'plot-rsa-table':
        P.plot_saved_similarity_matrix(subject_id=args.subject_id, task_id=args.task_id,word_pos=args.word_pos.split(","), sort_order=args.sort_order)
    elif args.command == 'plot-bar-average-rsa':
        correlations, noise_ceiling = rsa._compare_with_models_subjects(models=["GLOVE", "BERT"], word_pos=args.word_pos.split(","))
        P.plot_average_rsa_from_correlations(correlations, noise_ceiling_map = noise_ceiling)
    # TODO : per-electrode topographic map with time.
    # TODO : spliding-window by nouns vs verbs.
    else:
        print("Please provide a valid command ('compare' or 'generate').")


if __name__ == "__main__":
    main()
