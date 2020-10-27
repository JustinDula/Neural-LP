import os
import re
import random
from os.path import join as pjoin
from shutil import rmtree

NUM_FOLDS = 5
FOLDS = ["datasets/5folds/fold" + str(i + 1) for i in range(NUM_FOLDS)]
OUTPUT_FOLDS = ["datasets/5folds-processed/fold" + str(i + 1) for i in range(NUM_FOLDS)]
COAUTHOR_TUPLES = []
AUTHORS = set()

# RANDOM_SEED = 214812934
# random.seed(RANDOM_SEED)


def make_entity_files():
	auth_string = "\n".join(AUTHORS) + "\n"
	for path in OUTPUT_FOLDS:
		with open(pjoin(path, "entities.txt"), "w") as f:
			f.write(auth_string)


def make_relations_files():
	for path in OUTPUT_FOLDS:
		os.makedirs(path)
		with open(pjoin(path, "relations.txt"), "w") as f:
			f.write("CoAuthor\n")


def tuples_to_relation_strings(l):
	str_list = [a + "\tCoAuthor\t" + b for a,b in l]
	return "\n".join(str_list) + "\n"


def random_split(l, p):
	a, b = [], []
	for x in l:
		if random.random() < p:
			a.append(x)
		else:
			b.append(x)


def make_test_train_files():
	for i in range(NUM_FOLDS):
		path = OUTPUT_FOLDS[i]
		test_p, test_n = COAUTHOR_TUPLES[i]
		train_p, train_n = [], []

		for pos, neg in COAUTHOR_TUPLES[:i] + COAUTHOR_TUPLES[i+1:]:
			train_p += pos
			train_n += neg

		random.shuffle(train_p)
		split = int(len(train_p) * .75)
		facts_p, train_p = train_p[:split], train_p[split:]

		with open(pjoin(path, "facts.txt"), "w") as f:
			f.write(tuples_to_relation_strings(facts_p))

		with open(pjoin(path, "train.txt"), "w") as f:
			f.write(tuples_to_relation_strings(train_p))

		with open(pjoin(path, "test.txt"), "w") as f:
			f.write(tuples_to_relation_strings(test_p))


def process_folds():
	global FOLDS, OUTPUT_FOLDS, COAUTHOR_TUPLES, AUTHORS

	try:
		rmtree("datasets/5folds-processed")
	except OSError:
		pass

	raw_pattern = r"""CoAuthor\("([\w_]+)","([\w_]+)"\)\."""
	re_pattern = re.compile(raw_pattern)

	for in_path in FOLDS:
		neg = []
		with open(pjoin(in_path, "neg.txt"), "r") as f:
			for line in f.readlines():
				if line.strip() == "":
					pass
				match = re_pattern.match(line).groups()
				neg.append(match)
				AUTHORS.update(match)
		pos = []
		with open(pjoin(in_path, "pos.txt"), "r") as f:
			for line in f.readlines():
				if line.strip() == "":
					pass
				match = re_pattern.match(line).groups()
				pos.append(match)
				AUTHORS.update(match)

		COAUTHOR_TUPLES.append((pos, neg))

	AUTHORS = list(AUTHORS)

	make_relations_files()
	make_entity_files()
	make_test_train_files()


def main():
	process_folds()


if __name__ == "__main__":
	main()
