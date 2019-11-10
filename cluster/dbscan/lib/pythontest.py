# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-25 18:37:41
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-27 16:18:37

import time
import ctypes  
import time

if __name__ == '__main__':
	start = time.time()
	ll = ctypes.cdll.LoadLibrary   
	lib = ll("./testc.so")  
	lib.fibonacci(40)
	end = time.time()
	print(end - start)   # 0.4253208637237549
