#!/bin/env python3
import os

projectPath = os.path.dirname(os.path.realpath(__file__))
datasetDir = os.path.realpath(projectPath + "/../cifar-10-batches-py")
batchBasePath = datasetDir + "/data_batch_"

# todo: remove those useless config entries