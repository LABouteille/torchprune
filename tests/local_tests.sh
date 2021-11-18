#!/bin/sh

python -m pytest pruner/test_dependency_graph.py -s -vv
python -m pytest pruner/test_criteria.py -s -vv
python -m pytest pruner/test_pruner_conv.py -s -vv
python -m pytest pruner/test_pruner_linear.py -s -vv
python -m pytest pruner/test_pruner_conv_linear.py -s -vv
python -m pytest pruner/test_pruner_batchnorm.py -s -vv
python -m pytest pruner/test_pruner.py -s -vv
#python -m pytest pruner/test_pruner_skip_connection.py -s -vv
