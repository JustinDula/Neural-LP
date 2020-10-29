import os
import re
import random
from os.path import join as pjoin
from shutil import rmtree

NUM_FOLDS = 5
FOLDS = ["datasets/5folds/fold" + str(i + 1) for i in range(NUM_FOLDS)]
OUTPUT_FOLDS = ["datasets/5folds-processed/fold" + str(i + 1) for i in range(NUM_FOLDS)]
COAUTHOR_TUPLES = []
ENTITIES = set()
RELATIONS = {"CoAuthor"}
FACTS = list()

# RANDOM_SEED = 214812934
# random.seed(RANDOM_SEED)


def make_entity_files():
	auth_string = "\n".join(ENTITIES) + "\n"
	for path in OUTPUT_FOLDS:
		with open(pjoin(path, "entities.txt"), "w") as f:
			f.write(auth_string)


def make_relations_files():
	for path in OUTPUT_FOLDS:
		os.makedirs(path)
		with open(pjoin(path, "relations.txt"), "w") as f:
			f.write("\n".join(RELATIONS) + "\n")


def tuples_to_relation_strings(l):
	str_list = [a + "\tCoAuthor\t" + b for a, b in l]
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
			for rel, a, b in FACTS:
				f.write("{}\t{}\t{}\n".format(a, rel, b))

		with open(pjoin(path, "train.txt"), "w") as f:
			f.write(tuples_to_relation_strings(train_p))

		with open(pjoin(path, "test.txt"), "w") as f:
			f.write(tuples_to_relation_strings(test_p))
			f.write(tuples_to_relation_strings(test_n))

		labels = ([1] * (len(test_p) * 2)) + ([0] * (len(test_n) * 2))

		with open(pjoin(path, "labels.txt"), "w") as f:
			f.write("\n".join(str(x) for x in labels) + "\n")


def process_folds():
	global FOLDS, OUTPUT_FOLDS, COAUTHOR_TUPLES, ENTITIES, RELATIONS

	try:
		rmtree("datasets/5folds-processed")
	except OSError:
		pass

	raw_pattern = r"""([\w]+)\("([\w_]+)","([\w_]+)"\)\."""
	re_pattern = re.compile(raw_pattern)

	with open("datasets/5folds/train_facts.txt") as f:
		for line in f:
			match = re_pattern.match(line.strip())
			if match:
				rel, a, b = match.groups()
				RELATIONS.add(rel)
				ENTITIES.update((a,b))
				FACTS.append((rel, a, b))

	for in_path in FOLDS:

		neg = []
		with open(pjoin(in_path, "neg.txt"), "r") as f:
			for line in f.readlines():
				match = re_pattern.match(line)
				if match:
					rel, a, b = match.groups()
					neg.append((a, b))
					ENTITIES.update((a, b))

		pos = []
		with open(pjoin(in_path, "pos.txt"), "r") as f:
			for line in f.readlines():
				match = re_pattern.match(line)
				if match:
					rel, a, b = match.groups()
					pos.append((a, b))
					ENTITIES.update((a, b))

		COAUTHOR_TUPLES.append((pos, neg))

		COAUTHOR_TUPLES.append((pos, neg))

	make_relations_files()
	make_entity_files()
	make_test_train_files()


def main():
	process_folds()


if __name__ == "__main__":
	main()
