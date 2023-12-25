# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:48:30 2022
@author: Yibin Ye
"""

from super_utils import *

if __name__ == '__main__':
    Your_path = ''
    save_dir = Your_path + '/MFAMatch/training_validation_set/'
    whole_Path = Your_path + '/MFAMatch/train_validation_data'
    n = 10
    read_and_label(whole_Path, save_dir, n)