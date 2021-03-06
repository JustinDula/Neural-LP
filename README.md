# Nueral LP for 5folds

Use `run.py` with an a appropriate virtual environment to run with the 5 folds 
given in `datasets/5folds`.  

The output is given in `coauthor_results.txt`. 
A link prediction is treated as true iff the most probable completion
matches the true completion.

# Author Readme
# Neural LP

This is the implementation of Neural Logic Programming, proposed in the following paper:

[Differentiable Learning of Logical Rules for Knowledge Base Reasoning](https://arxiv.org/abs/1702.08367).
Fan Yang, Zhilin Yang, William W. Cohen.
NIPS 2017.

## Requirements
- Python 2.7
- Numpy 
- Tensorflow 1.0.1
- Scikit-learn 0.20.4

## Quick start
The following command starts training a dataset about family relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo
```

Wait for around 8 minutes, navigate to `exps/demo/`, there is `rules.txt` that contains learned logical rules. 

## Evaluation
To evaluate the prediction results, follow the steps below. The first two steps is preparation so that we can compute _filtered_ ranks (see [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) for details).

We use the experiment from Quick Start as an example. Change the folder names (datasets/family, exps/dev) for other experiments.
```
. eval/collect_all_facts.sh datasets/family
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds=exps/demo/test_predictions.txt --truths=datasets/family/truths.pckl
```
