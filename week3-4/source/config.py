#!/bin/env python3
import os

projectRoot = os.path.dirname(os.path.realpath(__file__))
datasetDir = os.path.realpath(projectRoot + "/../cifar-10-batches-py")
batchBasePath = datasetDir + "/data_batch_"
