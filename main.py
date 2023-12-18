import minimatrix



mat_data = [[1,2,3],[4,5,6],[7,8,9]]
mat = minimatrix.Matrix(data=mat_data)
print(mat)

m24 = minimatrix.arange(0,24,1)
print(m24)

# print(m24.reshape((3,8)),m24.reshape((24,1)),m24.reshape((4,6)))

# print(minimatrix.zeros((3,3)),minimatrix.zeros_like(m24))

# print(minimatrix.ones((3,3)),minimatrix.ones_like(m24))

# print(minimatrix.nrandom((3,3)),minimatrix.nrandom_like(m24))


def least_squares_method(m,n):
    X = minimatrix.nrandom((m,n))
    w = minimatrix.nrandom((n,1))
    e = minimatrix.nrandom((m,1))
    average_e = e.sum()[0,0]/m
    e = e - minimatrix.Matrix(dim=(m,1),init_value=average_e) #零均值化
    Y = X.dot(w) + e 
    print(0)
    w_ = (X.T().dot(X)).inverse().inverse().dot(X.T().dot(Y)) 
    return w ,w_

print(least_squares_method(1000,100))