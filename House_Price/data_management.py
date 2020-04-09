# -*- coding: utf-8 -*-
"""
data_management.py
"""

import pandas as pd

def load_dataset( file_name ):
    
    data = pd.read_csv( file_name )
    
    return data