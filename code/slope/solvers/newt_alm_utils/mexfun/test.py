# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:11:55 2022

@author: Prince_Li
"""

from mexMatvec123 import *
import numpy as np

A = np.random.rand(3,2)
b = np.random.rand(2,1)

mexMatvec123(A,b,0)