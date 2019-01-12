import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("set CUDA device!!!")

print('importing packages...')
from paint import *

global m_paint
m_paint=paint()

m_paint.load('face3.png')

m_paint.r(20)
index = m_paint.index.reshape(m_paint.index.shape[0] * m_paint.index.shape[1]).astype(int)
indextolist = index.tolist()
index_set = set(indextolist)
index_list=list(index_set)
print("index size before repaint: ", len(index_list)-1)
#m_paint.savestroke(s_list=index_list)


from collections import Counter
stroke_count=Counter(indextolist)
stroke_count_value=list(stroke_count.values())
stroke_count_value.sort()
print(stroke_count_value)
for key in list(stroke_count.keys()):
    #print(key,stroke_count[key])
    if stroke_count[key]< 10:
        del stroke_count[key]
print(len(stroke_count))

stroke_count_key = list(stroke_count.keys())
stroke_count_key.sort()
m_paint.savestroke(s_list=stroke_count_key)

cv2.waitKey(0)

