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


def compute_all_rsa_matrics(task_id = None):
    subject_ids = D.load_subject_ids()
    task_ids = D.load_task_ids() if not task_id else [task_id] 

    for subject_id in subject_ids:
        for task_id in task_ids: 
            word_index, similarity_matrix_0 = \
                rsa._get_similarity_matrix(subject_id=subject_id, task_id=task_id, save_similarity_matrix=True)

def compute_rsa_matrix(subject_id, task_id):
    word_index, similarity_matrix_0 = rsa._get_similarity_matrix(subject_id=subject_id, task_id=task_id, save_similarity_matrix=True)

def main():
    # Parse arguments
    # Initialize the main parser
    parser = argparse.ArgumentParser(description='Perform RSA analysis.')
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')

    # Subcommand for comparing two subjects
    compare_parser = subparsers.add_parser('compare', help='Compare RSA similarity between two subjects')
    compare_parser.add_argument('--subject1', type=str, required=True, help='ID of the first subject')
    compare_parser.add_argument('--subject2', type=str, required=True, help='ID of the second subject')
    compare_parser.add_argument('--task_id', type=int, required=True, help='ID of the task to compare')

    # Subcommand for generating similarity matrix for one subject
    generate_parser = subparsers.add_parser('generate', help='Generate similarity matrix for a subject')
    generate_parser.add_argument('--subject', type=str, required=True, help='ID of the subject')
    generate_parser.add_argument('--task_id', type=int, required=True, help='ID of the task')

    generate_parser = subparsers.add_parser('generate-all', help='Generate all similarity matrics for all tasks.')

    args = parser.parse_args()

    if args.command == 'compare':
        _compare_subjects(args.subject1, args.subject2, task_id=args.task_id)
        correlation = compute_rsa_similarity(args.subject1, args.subject2, args.task_id)
        # Output the result
        print(f"RSA correlation between subject {args.subject1} and {args.subject2} for task {args.task_id}: {correlation:.2f}")
    elif args.command == 'generate':
        compute_rsa_matrix(subject_id=args.subject, task_id=args.task_id)
    elif args.command == 'generate-all':
        compute_all_rsa_matrics()
    else:
        print("Please provide a valid command ('compare' or 'generate').")


if __name__ == "__main__":
    main()


