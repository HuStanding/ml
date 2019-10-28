# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-25 18:37:41
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-25 18:38:05

import time
from numba import jit
import pandas as pd

@jit()
def use_pandas(a):
    df = pd.DataFrame.from_dict(a)
    df += 1 
    return df.cov()       

if __name__ == '__main__':
	x = {'a': [1, 2, 3], 'b': [20, 30, 40]}
	start = time.time()
	use_pandas(x)
	end = time.time()
	print(end - start)     # 0.7431411743164062

	start = time.time()
	use_pandas(x)
	end = time.time()
	print(end - start)     # 0.002971172332763672