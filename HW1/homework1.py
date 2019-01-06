import numpy as np
import time
import random

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime = time.time()
z_1 = [[0 for x in range(5)] for y in range(3)]
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3,5),dtype=int)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))



#################################################
# 2. Set all the elements in first row of z to 7.
# Python
pythonStartTime = time.time()
z_1[0] = [7]*5;
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
z_2[0:1,:] = 7
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
pythonStartTime = time.time()
z_1[0][1] = 9
z_1[1][1] = 9
z_1[2][1] = 9
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
z_2[:,1:2] = 9
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
numPyStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
z_2[1:2,2:3] = 5
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime = time.time()
x_1 = [i for i in range(50,100)]
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50,100)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python

def createMatrix(rowC, colC, listV):
    listmatrix = []
    for i in range(rowC):
        listRow = []
        for j in range(colC):
            # you need to increment through dataList here, like this:
            listRow.append(listV[colC * i + j])
        listmatrix.append(listRow)

    return listmatrix

pythonStartTime = time.time()
x_temp = [i for i in range(16)]
y_1 = createMatrix(4,4,x_temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
y_2 = np.array(np.arange(0,16)).reshape((4,4))
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [[0 for i in range(5)] for j in range(5)]
tmp_1[0] = [1]*5
tmp_1[4] = [1]*5

tmp_1[1][0] = 1
tmp_1[1][4] = 1

tmp_1[2][0] = 1
tmp_1[2][4] = 1

tmp_1[3][0] = 1
tmp_1[3][4] = 1
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5,5), dtype='int')
tmp_2[1:-1,1:-1] = 0
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
pythonStartTime = time.time()
x_temp = [i for i in range(5000)]
a_1 = createMatrix(50,100,x_temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
a_2 = np.array(np.arange(0,5000).reshape((50,100)))
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
x_temp = [i for i in range(20000)]
b_1 = createMatrix(100,200,x_temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))


# NumPy
numPyStartTime = time.time()
b_2 = np.array(np.arange(0,20000).reshape((100,200)))
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
pythonStartTime = time.time()
c_1 = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*b_1)] for X_row in a_1]
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
c_2 = np.dot(a_2,b_2)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
pythonStartTime = time.time()

x_temp = random.sample(range(10), 9)
x_min = min(x_temp)
x_max = max(x_temp)
temp_list = [((x-x_min)/(x_max-x_min)) for x in x_temp]
d_1 = createMatrix(3,3,temp_list)

pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
d_2 = np.random.randint(10,size=(3,3))
d_2_max = d_2.max()
d_2_min = d_2.min()
d_2 = (d_2-d_2_min)/(d_2_max-d_2_min)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##########
print(d_1)
print(d_2)
#########

################################################
# 12. Subtract the mean of each row of matrix a.
# Python
pythonStartTime = time.time()
tmp_v = 0
newList = []
for i in range(50):
    sum =0
    for j in range(tmp_v, tmp_v+100):
        sum = sum+j
    eachrow_mean = sum/100

    for j in range(tmp_v, tmp_v+100):
        newList.append((j-eachrow_mean))
    tmp_v = tmp_v+100

a_1 = createMatrix(50,100,newList)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
a_2 = (a_2 - a_2.mean(axis =1, keepdims=True))
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
pythonStartTime = time.time()
tmp_v = 0
newList2 = []
for i in range(100):
    eachcol_mean = 9900
    for j in range(tmp_v, tmp_v+200):
        newList2.append((j-eachcol_mean))
        eachcol_mean += 1
    tmp_v = tmp_v+200

b_1 = createMatrix(100,200,newList2)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
b_2 = (b_2 - b_2.mean(axis =0, keepdims=True))
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
pythonStartTime = time.time()
c_1= list(map(list, zip(*c_1)))
c_1 = [j for i in c_1 for j in i]
c_1 = [(x+5) for x in c_1]
e_1 = createMatrix(200,50,c_1)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
e_2 = c_2.transpose();
e_2 += 5
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##################
print("The sum for e_1 and e_2 element values does not come to be equal because c_1 and c_2 are not equal as "
      "we have used zip in python and dot in numpy for matrix multiplication!")
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
pythonStartTime = time.time()
f_1 = [j for i in e_1 for j in i]
formatStr = '({0}, 1)'.format(f_1.__len__())
print (formatStr)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
f_2 = e_2.flatten()
print(f_2.shape)
numPyEndTime = time.time()
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))