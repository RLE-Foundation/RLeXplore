#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：__init__.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 13:51 
'''
from rlexplore.utils.envs import create_env
from rlexplore.utils.state_process import process

import os, glob
def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)