#!/bin/sh

#python -m pytest test_dependency_graph.py -s -vv
#python -m pytest test_criteria.py -s -vv
#python -m pytest test_pruner_conv.py -s -vv
#python -m pytest test_pruner_linear.py -s -vv
python -m pytest test_pruner_conv_linear.py -s -vv
python -m pytest test_pruner.py -s -vv
