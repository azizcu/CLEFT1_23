import pdb
import logging
import argparse
import os

import sys
sys.path.append('.')
from format_checker.subtask_1 import check_format
from scorer.utils import _compute_average_precision, _compute_reciprocal_rank, _compute_precisions
from scorer.utils import print_thresholded_metric, print_single_metric
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
Scoring of Task 1 with the metrics accuracy, and precision, recall, and f1 for positive class.
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50]

def _read_gold_and_pred(gold_fpath, pred_fpath):
    """
    Read gold and predicted data.
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number and score at each line.
    :return: {line_number:label} dict; list with (line_number, score) tuples.
    """

    logging.info("Reading gold labels from file {}".format(gold_fpath))

    gold_labels = {}
    with open(gold_fpath, encoding='utf-8') as gold_f:
        next(gold_f)
        for line_res in gold_f:
            (id, tweet_text, label) = line_res.strip().split('\t')  # process the line from the res file
            gold_labels[str(id)] = label

    logging.info('Reading predicted labels from file {}'.format(pred_fpath))

    line_score = []
    # labels=[]
    with open(pred_fpath) as pred_f:
        next(pred_f)
        for line in pred_f:
            id, pred_label, run_id  = line.split('\t')
            id = str(id.strip())
            pred_label = pred_label.strip()

            if id not in gold_labels:
                logging.error('No such id: {} in gold file!'.format(id))
                quit()
            line_score.append((id, pred_label))
            # labels.append(pred_label)


    if len(set(gold_labels).difference([tup[0] for tup in line_score])) != 0:
        logging.error('The predictions do not match the lines from the gold file - missing or extra line_no')
        raise ValueError('The predictions do not match the lines from the gold file - missing or extra line_no')

    return gold_labels, line_score


def evaluate(gold_fpath, pred_fpath):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    """

    #Gold files are differently formatted for 1A and 1B, we need to update this part to serve both
    gold_labels_dict, pred_labels_dict = _read_gold_and_pred(gold_fpath, pred_fpath)
    gold_labels=[]
    pred_labels=[]
    for t_id, label in gold_labels_dict.items():
        gold_labels.append(label)
    for t_id, label in pred_labels_dict:
        pred_labels.append(label)

    # Calculate Metrics
    acc = accuracy_score(gold_labels, pred_labels)
    precision = precision_score(gold_labels, pred_labels, pos_label='Yes',average='binary')
    recall = recall_score(gold_labels, pred_labels, pos_label='Yes',average='binary')
    f1 = f1_score(gold_labels, pred_labels, pos_label='Yes',average='binary')

    return acc, precision, recall, f1


def validate_files(pred_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
    
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to file with predict class per tweet.")
    # parser.add_argument("--subtask", "-a", required=True,
    #                     choices=['A', 'B'],
    #                     help="The subtask you want to score its runs.")
    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path
    # subtask = args.subtask

    if validate_files(pred_file):
        logging.info(f"Started evaluating results for subtask-1B-English ...")
        overall_precisions = [0.0] * len(MAIN_THRESHOLDS)

        acc, precision, recall, f1 = evaluate(gold_file, pred_file)
        print("acc: {}, P:{}, R:{}, F1:{}".format(acc, precision, recall, f1))

