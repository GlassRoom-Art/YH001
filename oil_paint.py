print('importing packages...')
from paint import *

global m_paint
m_paint=paint()

m_paint.load('baby.jpg')

m_paint.r(2)
index = m_paint.index.reshape(m_paint.index.shape[0] * m_paint.index.shape[1]).astype(int)
indextolist = index.tolist()
index_set = set(indextolist)
index_list=list(index_set)
print("index size before repaint: ", len(index_list)-1)
m_paint.savestroke(s_list=index_list)

from collections import Counter
stroke_count=Counter(indextolist)
for key, value in stroke_count.items():
    print(key,value)

cv2.waitKey(0)

