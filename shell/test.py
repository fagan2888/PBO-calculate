
import pandas as pd
from itertools import product, permutations, combinations, combinations_with_replacement
import math
import numpy as np
from ipdb import set_trace

## 列出所有可能，有放回。有顺序的结果
#result = product('abc',repeat = 2)
#print(list(set(result))[1])
#
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

print(math.log(10))
set_trace()
df  = pd.DataFrame({'a':[1,2], 'b':[2,3.5], 'c':[3.6,4]})
mean = df.mean()
std = df.std()
print(df)
print(mean)
print(std)
position = mean.sort_values(ascending = False).rank().index[0]
print(position)
rank = std.rank()[position]
print(rank)


