
import pandas as pd
from itertools import product, permutations, combinations, combinations_with_replacement
import math
import numpy as np
from ipdb import set_trace

# 列出所有可能，有放回。有顺序的结果
result = product('1234567890abcdes',repeat = 8)
print(len(set(result)))

## 列出所有可能，无放回，有顺序的结果（也即排列）
#result = permutations('abc', 2)
#print(set(result))
#
## 列出所有可能，无放回，无顺序的结果（也即组合）
#result = combinations('abc', 2)
#print(set(result))
#
## 列出所有可能，有放回，无顺序的结果
#result = combinations_with_replacement('abc', 2)
#print(set(result))
#



