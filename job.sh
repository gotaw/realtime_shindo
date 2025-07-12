#!/bin/bash

#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -g gv49
#PJM -j

#------- Environment Setup -------#
source /work/gv49/e15000/miniconda3/etc/profile.d/conda.sh
conda activate /work/04/gv49/e15000/gotaw/realtime_shindo/.conda

#------- Program execution -------#
# python realtime_shindo/modeling/train.py
# python realtime_shindo/modeling/predict.py -v $(ls -d models/lightning_logs/version_* | sort -V | tail -n 1)