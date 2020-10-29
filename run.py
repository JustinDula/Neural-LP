from __future__ import print_function

import os
from shutil import rmtree
from collections import namedtuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from process_folds import process_folds

SETUP = "python2 src/process_folds.py"
COMMANDS = [
	"python2 src/main.py --datadir=datasets/5folds-processed/fold{n} --exps_dir=exps/ --exp_name=fold{n}"
	" --max_epoch=100 --get_phead",
	". eval/collect_all_facts.sh datasets/5folds-processed/fold{n}",
	"python eval/get_truths.py datasets/5folds-processed/fold{n}",
	"python eval/evaluate.py --preds=exps/fold{n}/test_predictions.txt"
	" --truths=datasets/5folds-processed/fold{n}/truths.pckl"
	" | tee exps/fold{n}/eval.txt"
]

PREDICTION = "exps/fold{n}/test_preds_and_probs.txt"
THRESHOLD = .2


def rerun_experiments():
	print(os.getcwd())
	for i in range(5):

		try:
			rmtree("exps/fold" + str(i + 1))
		except OSError:
			pass
		os.makedirs("exps/fold" + str(i+1))
		for command in COMMANDS:
			os.system(command.format(n=i + 1))


def evaluate():
	PredictionInfo = namedtuple("PredictionInfo", "true_labels, pred_label, matches, probs, test_size")

	# auc roc, auc pr, precision, recall, f1
	true_labels_per_fold = []
	results = []
	for i in range(5):
		path = PREDICTION.format(n=i+1)
		matches = []
		probs = []
		with open(path, "r") as f:
			for line in f.readlines():
				l = line.split(",")
				match = l[1] == l[3]
				p = float(l[-1])
				pred_label = 1 if p >= THRESHOLD else 0
				matches.append(match)
				probs.append(p)

		with open("datasets/5folds-processed/fold{n}/labels.txt".format(n=i+1), "r") as f:
			labels = [int(x) for x in f.readlines() if x.strip() != ""]

		results.append(PredictionInfo(true_labels=labels, matches=matches, probs=probs))

	results = {
		"auc roc": [roc_auc_score(p.true_labels, p.probs) for p in results],
		"auc pr": [average_precision_score(p.true_labels, p.probs) for p in results],
		"precision": [precision_score(p.true_labels, p) for p in results],
		"recall": [recall_score(p.true_labels, l) for p in results],
		"f1s": [f1_score(p.true_labels, p.pred_label) for p in results]
	}

	print("label, [scores], mean, std")
	for k, v in results.items():
		print(k, v, np.average(v), np.std(v))

	with open("5folds_results.txt", "w") as f:
		f.write("label, [scores], mean, std\n")
		for k, v in results.items():
			f.write("{} {} {} {}\n".format(k, v, np.average(v), np.std(v)))


def main():
	# process_folds()
	# rerun_experiments()
	evaluate()


if __name__ == '__main__':
	main()
