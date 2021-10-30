#!/bin/sh

pytest test_dependency_graph.py -s -vv
pytest test_pruner.py -s -vv
