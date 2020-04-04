from cv2 import 
cap = cv2.VideoCapture(0)    #调用笔记本内置摄像头

words = 'hello, cat'
words[::-1]
'%s,%s,%d'%('hello','lorry',5)
words[-4:-2]

A=[1,2,5,6,9]
B=[2,8,9,11]
print(set(A)&set(B))
print(set(A)^set(B))


def getInt():
    r = range(1, 100)
    w = range(0.1, 0.9)
    for i in r:
        yield from w
    # yield from r

for i in getInt():
    print(i)

