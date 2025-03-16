import numpy as np


# landmark = 0

# inf = 10

# def ZLJ(landmark):
#     landmark = landmark + 1
#     if landmark == inf:
#         return "ZLJ's"
#     return str(ZLJ(landmark)) +" Landmark Journey"

# if __name__ == '__main__':
#     print(ZLJ(landmark))

# x = np.array([[[ 3.14178734e-05+6.11099482e-10j, -5.28504900e-05+3.39530134e-05j]], [[-1.93842311e-05-1.15504160e-04j,4.76717032e-05+1.07039255e-04j]]])
# print(x.shape)
# sum_H = np.sum(x, axis=0)
# print(sum_H.shape)


x = np.array([[0, 1, 2], 
              [3, 4, 5],
              [6, 10, 8],
              [9, 9, 11]])

print(x[:, 1])
e = np.argmax(x[:, 1])
print(e)

print(x[e, 1])