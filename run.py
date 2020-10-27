from __future__ import print_function

import os
from shutil import rmtree

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from process_folds import process_folds

"""
Names are ranked in 
"""

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


def rerun_experiments():
	process_folds()

	print(os.getcwd())
	for i in range(5):

		try:
			rmtree("exps/fold" + str(i + 1))
		except OSError:
			pass

		for command in COMMANDS:
			os.system(command.format(n=i + 1))


def evaluate():
	# auc roc, auc pr, precision, recall, f1

	results = []
	for i in range(5):
		path = PREDICTION.format(n=i+1)
		labels = []
		probs = []
		with open(path, "r") as f:
			for line in f.readlines():
				l = line.split(",")
				labels.append(1 if l[1] == l[3] else 0)
				probs.append(float(l[-1]))
		results.append((labels, probs))

	results = {
		"auc roc": [roc_auc_score(l, p) for l, p in results],
		"auc pr": [average_precision_score(l, p) for l, p in results],
		"precision": [precision_score([1 for _ in l], l) for l, _ in results],
		"recall": [recall_score([1 for _ in l], l) for l, _ in results],
		"f1s": [f1_score([1 for _ in l], l) for l, _ in results]
	}

	print("label, [scores], mean, std")
	for k, v in results.items():
		print(k, v, np.average(v), np.std(v))

	with open("results.txt", "w") as f:

		f.write("label, [scores], mean, std\n")
		for k, v in results.items():
			f.write("{} {} {} {}\n".format(k, v, np.average(v), np.std(v)))


def main():
	# rerun_experiments()
	evaluate()


if __name__ == '__main__':
	main()
