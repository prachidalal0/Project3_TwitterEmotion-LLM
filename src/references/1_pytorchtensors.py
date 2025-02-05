# -*- coding: utf-8 -*-
"""1_PyTorchTensors.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15opNVR3UrWQJ8m5LnP30Bbja9eg7UtsB

# <font color = 'pickle'>**Lecture : Intoduction to PyTorch Tensors**

# <font color = 'pickle'>**Importing PyTorch Library**
"""

!nvidia-smi

import torch
import numpy as np

"""# <font color = 'pickle'>**Tensors**

- Tensors are the basic building blocks of any deep learning network.

- They are used to represent all the different types of data be it images, sound files, text data etc.

- Tensors are **order N-matrix**.


If N=1, tensor will basically be a **vector**.
If N=2, tensor will be a **2-d matrix**.

Why Tensors and not NumPy arrays?

- NumPy only supports CPU computation.
- Tensor class supports automatic differentiation.

**Let us start by importing PyTorch library and understand some of the basic functions on tensors.**

## <font color = 'pickle'>**Scalar**
- rank-0 tensor
"""

t = torch.tensor(1.)
print(t)
print("\nsize:",t.shape, sep ='\n')
print("\nnumber of dimensions:", t.dim(), sep = "\n")
print("\nData Type:", t.dtype, sep = "\n")

"""## <font color = 'pickle'>**Vector**
- rank-1 tensor
"""

t = torch.tensor([1., 2])
print(t)
print("\nsize:",t.shape, sep ='\n')
print("\nnumber of dimensions:", t.dim(), sep = "\n")
print("\nData Type:", t.dtype, sep = "\n")

"""## <font color = 'pickle'>**Matrix**
- rank 2 tensor

Matrices are 2-d arrays with size `n x m`. Here, n: number of rows and m: number of columns.

If `m = n`, then the matrix is known as a `square matrix`.

Precisely, matrices can be represented as:
$$\mathbf{X}=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \\ \end{bmatrix}$$
<br>


"""

t = torch.tensor([
     [1., 2, 3],
     [4, 5, 6]
    ])
print(t)
print("\nsize:",t.shape, sep ='\n')
print("\nnumber of dimensions:", t.dim(), sep = "\n")
print("\nData Type:", t.dtype, sep = "\n")

"""<img src = "https://drive.google.com/uc?export=view&id=1822fQJQuXtzZ7DmO86pUXUj4aU9ZLor_" width =600 >

## <font color = 'pickle'>**Higher Order Tensors**

### <font color = 'pickle'>**rank-3 tensor**
"""

t = torch.tensor([
    [[1 , 2], [3,4]],
    [[5, 6], [7,8]],
    [[5, 6], [7,8]]
                  ])
print(t)
print("\nsize:",t.shape, sep ='\n')
print("\nnumber of dimensions:", t.dim(), sep = "\n")
print("\nData Type:", t.dtype, sep = "\n")

"""<img src = "https://drive.google.com/uc?export=view&id=184X0Qjn0lwuJRSFoh_lEmR9v7yF7GxaA" width =600 >

Image source: https://dev.to/sandeepbalachandran/machine-learning-going-furthur-with-cnn-part-2-41km

### <font color = 'pickle'>**rank-4 tensor**
"""

t1 = torch.stack((t,t))
print(t1)
# print(t)
print("\nsize:",t1.shape, sep ='\n')
print("\nnumber of dimensions:", t1.dim(), sep = "\n")
print("\nData Type:", t1.dtype, sep = "\n")

"""<img src = "https://drive.google.com/uc?export=view&id=189RzBY0oYuih-dZjNAT79FIVf3ZUp_HY" width =600 >

# <font color = 'pickle'> **Python list**
"""

scalar = 4
type(scalar)

my_list = [[1., 2], [3,4]]
type(my_list)

my_tensor = torch.tensor([[1., 2], [3,4]])
type(my_tensor)
print(t)
print("\nData Typr:", my_tensor.dtype, sep = "\n")

"""# <font color = 'pickle'>**Difference between list and Array/tensor**</font>

| <font size =5> Python List                       | <font size =5>Tensor/Array                     |
|-----------------------------------|----------------------------------|
| <font size =5>Mixed types allowed               | <font size =5>Same type required               |
|<font size =5> Elements can be added or removed  | <font size =5>Elements cannot be added or removed               
| <font size =5>Basic Python operations           | <font size =5>Supports mathematical operations                
|<font size =5>Numerical Computtaions are slow    |<font size =5>Numerical Computtaions are fast

# <font color = 'pickle'>**Conversion to other Python Objects**
"""

# Initializing a tensor
t = torch.arange(10)
print(t)

# Converting tensor t to numpy array using numpy() mehod
arr = t.numpy()

# Converting numpy array to tensor T using tensor() method
T = torch.tensor(arr)

# Printing data type of arr and T
print(type(arr), type(T), T.type(), sep='\n')

"""We can also use torch.from_numpy() and torch.as_tensor() to convert numpy array to PyTorch Tensor. However, with these methods, the PyTorch tensor and the source NumPy array share the same memory. This means that changes to one affect the other. However, the torch.tensor() function always makes a copy."""

my_ndarray = np.arange(10)
t_from_numpy = torch.from_numpy(my_ndarray)
t_as_tensor = torch.as_tensor(my_ndarray)
t_Tensor = torch.tensor(my_ndarray)

print(f"tensor craeted using torch.from_numpy before changing np array: {t_from_numpy}")
print(f"tensor craeted using torch.as_tensor before changing np array : {t_as_tensor}")
print(f"tensor craeted using torch.tensor before changing np array    : {t_Tensor}")

# change numpy array
my_ndarray[2] = 1000

print()
print(f"tensor craeted using torch.from_numpy after changing np array: {t_from_numpy}")
print(f"tensor craeted using torch.as_tensor after changing np array : {t_as_tensor}")
print(f"tensor craeted using torch.tensor after changing np array    : {t_Tensor}")

# Initializing a size-1 tensor
t = torch.tensor([10.5])

# Printing tensor
print(t)

# Accessing element of tensor using item function
# item returns the value of the tensor as python number
# works only for tensors with single element

print(t.item())

# we an also conver the tensor to python list
t = torch.tensor([10, 2])
print(t)
print(t.tolist())

"""# <font color = 'pickle'>**Changing Shape of Tensors**"""

t = torch.arange(10)
print(t)
print("\nsize:",t.shape, sep ='\n')

t = t.view(5,2)
print(t)
print("\nsize:",t.shape, sep ='\n')

t = t.view(-1,5)
print(t)
print("\nsize:",t.shape, sep ='\n')

"""# <font color = 'pickle'>**Changing datatype of Tensors**
When creating tensor we can pass the dtype as an argument. We can also change the datatype of tensors using to() and type() mehods. For a list of dtypes visit https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
"""

x = torch.tensor([8, 9, -3], dtype=torch.int)

# we can use type() method or to() method to change the datatype
print(f"Old: {x.dtype}")

# change the datatype to int64 using type() method
x = x.type(dtype=torch.int64)
print(f"New: {x.dtype}")

# change the datatype to int32 using t0() method
x = x.to(dtype=torch.int32)
print(f"Newer: {x.dtype}")

"""# <font color = 'pickle'>**Saving Memory - inplace operations**

In-place operation are operations that change the content of a given Tensor without making a copy.

Operations that have a `_` suffix are in-place. For example: `.add_()`. Operations like += or *= are also inplace operations.

We can also perform in-place opaeration usng the notation `Z[:] = <expression>`.

As in-place operations do not make a copy, they can save memory. However, we need to use them carefully. They can be problematic when computing derivatives because of an immediate loss of history. We will learn about derivatives and computation graphs in coming lectures.
"""

a = torch.tensor(10)
print(a)
print(id(a))
a += 1
print(a)
print(id(a))
a = a + 1
print(a)
print(id(a))

b = torch.tensor(10)
print(b)
print(id(b))
b.add_(1) # B += 1 memory efficient
print(b)
print(id(b))
b = b.add(1) # b = b +1 this is not
print(b)
print(id(b))

"""## <font color = 'pickle'>**1) Checking gpu**"""

torch.cuda.is_available()

# check if gpu is availaible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create a tensor
X = torch.tensor([1, 2, 3, 4])
x

# check the device attribute of the tensor
X.device

# move the tensor to gpu
X = X.to(device=0)

X.device

device

# it is more efficient to create the tensor on gpu directly
Y = torch.tensor([1, 2, 3], device=device)

# check the device attribute of the tensor
Y.device

"""## <font color = 'pickle'>**2) Memory allocation of in-place operations**"""

# create tensor
t1 = torch.randn(10000, 10000, device="cpu")

# move tensor to gpu
t1 = t1.to(device)
print(t1.device)

# we can use id() function to get memory location of tensor
print(f"initial memory location of tensor t1 is : {id(t1)}")

x = t1
print(f"initial memory location of x is : {id(x)}")

# Waits for everything to finish running
torch.cuda.synchronize()

# initial memory allocated
start_memory = torch.cuda.memory_allocated()

# inplace operation
t1 += 0.1
t1.add_(0.1)
# since the operation was inplace when we update t1 it will update x as well
print(x == t1)

print(f"final memory location of tensor t1 is: {id(t1)}")
print(f"final location of x is : {id(x)}")

# totall memory allocated after function call
end_memory = torch.cuda.memory_allocated()

# memory allocated because of function call
memory_allocated = end_memory - start_memory
print(memory_allocated / 1024**2)

"""From the above example wecan see that both x and t1 has same memory location. When we ue in-place operation on t1, it also updates x

## <font color = 'pickle'>**3) Memory allocation of out-of-place operations**
"""

# create tensor
t2 = torch.randn(10000, 10000, device="cpu")

# move tensor to gpu
t2 = t2.to(device)
print(t2.device)

# we can use id() function to get memory location of tensor
print(f"initial memory location of tensor t2 {id(t2)}")

y = t2
print(f"final memory location of y is : {id(y)}")

# Waits for everything to finish running
torch.cuda.synchronize()

# initial memory allocated
start_memory = torch.cuda.memory_allocated()

# out-place opertaions
t2 = t2 + 0.1

# since the operation was not inplace when we update t2 it will not update y
print(y == t2)

# we can use id() function to get memory location of tensor
print(f"final memory location of tensor t2 {id(t2)}")
print(f"final memory location of y is : {id(y)}")

# totall memory allocated after function call
end_memory = torch.cuda.memory_allocated()

# memory allocated because of function call
memory_allocated = end_memory - start_memory
print(memory_allocated / 1024**2)

"""From the above example we can see that initially both y and t2 has same memory location. After running t2 = t2 + 0.1, we will find that id(t2) points to a different location. That is because Python first evaluates t2 + 0.1, allocating new memory for the result and then makes t2 point to this new location in memory. Since we have not done in-place operation, updating t2 does not effect y. y still points to the same memory location.

# <font color = 'pickle'>**Linear Algebra**

## <font color = 'pickle'>**Dot product**
Dot product of 2 vectors x and y  is given by the summation of product of elements at the same position.

If we have 2 vectors x: [1, 2, 3, 4] and y: [1, 1, 2, 1]

(x.y) will be 1x1 + 2x1 + 3x2 + 4x1 = 13
"""

# Initializing 2 tensors
x = torch.Tensor([0, -1, 1, 0])
y = torch.Tensor([0, 1, 1, 0])

# Performing Dot product
torch.dot(x, y)

# Dot Product is equal to sum of products at the same position, thus the expression below will give similar result
torch.sum(x * y)

# Initializing 2 tensors
x = torch.Tensor([1, 0, 0, 1])
y = torch.Tensor([1, 0, 0, 1])

# Performing Dot product
torch.dot(x, y)

"""## <font color = 'pickle'>**Dot product vs. for Loop in Python**"""

import time
n = 1000000
a = torch.arange(n)
b = torch.arange(n)

def pytorch_dot(x, y):
    return x.dot(y)

# Commented out IPython magic to ensure Python compatibility.
# %timeit pytorch_dot(a,b)

x = [1,2]
y = [3,4]
list(zip(x,y))

a1 = a.tolist()
b1 = b.tolist()
def plain_python(x, y):
    output = 0
    for x_j, y_j in zip(x, y):
        output += x_j * y_j
    return output

# Commented out IPython magic to ensure Python compatibility.
# %timeit plain_python(a,b)

"""**Output 1:**

921 µs ± 182 µs per loop: This tells you that the code took an average of 921 microseconds (µs) to run for each loop, with a standard deviation of 182 microseconds. The standard deviation gives an indication of the variability in the timing across different runs, which can be affected by other processes running on the computer at the same time.

(mean ± std. dev. of 7 runs, 1000 loops each): This part provides details about how the timing was measured. The code was run 7 times, and each of those runs consisted of 1000 loops. The mean and standard deviation were calculated from these 7 runs.

**Output 2:**
6.78 s ± 392 ms per loop: This tells you that the code took an average of 6.78 seconds to run for each loop, with a standard deviation of 392 milliseconds. Since 1 second equals 1000 milliseconds, this standard deviation is less than half a second.

(mean ± std. dev. of 7 runs, 1 loop each): Similar to Output 1, this part tells you that the code was run 7 times, and each of those runs consisted of just 1 loop. The mean and standard deviation were calculated from these 7 runs.

**In comparison, Output 1 suggests a much faster execution time (in the order of microseconds) compared to Output 2 (in the order of seconds).**

## <font color = 'pickle'>**Operations on Metrices**
"""

# Creating 2 matrices

# First matrix
A = torch.arange(0, 25).reshape(5, 5)

# Second matrix : copy of A
B = A.clone()

print(A)
print(B)

"""### <font color = 'pickle'>**Addition of 2 matrices**"""

# Addition of 2 matrices
A + B

"""### <font color = 'pickle'>**Subtraction of 2 matrices**"""

# Subtraction of 2 matrices
A - B

"""### <font color = 'pickle'>**Multiplying Matrices with Scalars**"""

# Each element of matrix can be aded or multiplied by a scalar (broadcasting)
# This operation will not change the shape of a matrix or a Tensor
a = 2
print(a + A)
print()
print(a * A)

"""### <font color = 'pickle'>**Transpose of a Matrix**"""

# Transpose of a matrix : Elements of the rows and columns get interchanged a[i][j] becomes a[j][i]
# Transpose is a special case of permute
A.T

"""### <font color = 'pickle'>**Hadamard product**"""

# Elementwise multiplication of two metrices is called Hadamard product
A * B

"""### <font color = 'pickle'>**Matrix Multiplication**

Matrix multiplication is a binary operation on 2 matrices which gives us a matrix which is the product of the 2 matrices.

If we are given 2 matrices $A$ of shape $(m * n)$ and $B$ of shape $(q * p)$, **we can perform matrix multiplication only when $n = q$** and the resultant product matrix will have shape $(m * p)$.

Suppose we are given 2 matrices $A (m * n)$ and $B (n * p)$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1n} \\
 a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1p} \\
 b_{21} & b_{22} & \cdots & b_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{n1} & b_{n2} & \cdots & b_{np} \\
\end{bmatrix}$$

Then after performing matrix multiplication, the resultant matrix C = AB will be:

$$\mathbf{C}=\begin{bmatrix}
 c_{11} & c_{12} & \cdots & c_{1p} \\
 c_{21} & c_{22} & \cdots & c_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
 c_{mp} & c_{mp} & \cdots & c_{mp} \\
\end{bmatrix}$$

Here, $c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + ... a_{in}b_{b_nj} = \sum_{k = 1}^n a_{ik}b_{kj}$

for, $i = 1,....m$ and $j = 1,...p$


Thus, each element of C, $c_{ij}$ is obtained by dot product of $i^{th}$ row of $A$ and $j^{th}$ column of $B$.

**Example** :
  1. Let $A$ be a matrix of (4, 3) dimensions.
  2. Let $B$ be another matrix of (3, 2) dimensions.
  3. Let us denote denote the matrix multiplication of $A$ and $B$ with $C$.
  5. Then the dimension of $C$ = (number of rows of $A$,number of columns of $B$)
     
    dimension of  $C$ = (4, 2)

The figure given below will give a good example of matrix multplication :

<img src = "https://drive.google.com/uc?view=export&id=176DF50XdtwkqU5wvxtWuD75sRDHvSJBf" width ="250"/>

We can perform matrix multiplication in the following way using PyTorch:
"""

# Initializing 2 matrices
A = torch.arange(0, 10, dtype=float).reshape(2, 5)
B = torch.ones(5, 2, dtype=float)

# Matrix-Matrix Multiplication using mm function of PyTorch
torch.mm(A, B)

A @ B

A * B

"""#### <font color = 'pickle'>**Prediction on Multiple Training Examples via Matrix Multiplication**"""

bias = torch.tensor([0.])
theta = torch.tensor([0.2, 9.3])
theta = theta.view(-1,1)
X = torch.tensor(
   [[1.8, 9.2],
    [0.2, 3.3],
    [5.2, 3.4],
    [3.4, 4.5],
    [6.1, 7.1]]
)
print(X.shape, theta.shape, bias.shape)

predictions = X.matmul(theta) + bias
predictions

"""# <font color = 'pickle'>**Self Study**

## <font color = 'pickle'>**Broadcasting - Operations on tensors of different  size**

Broadcasting describes how a tensor has to be treated during arithematic operation. If we have tensors of different sizes, we can broadcast the smaller array across the larger one so that they can have comaptible shapes.

### <font color = 'pickle'>**Broadcasting Examples**</font>

#### <font color = 'pickle'>**Broadcasting with a scalar**</font>
"""

t= torch.tensor([1,-2, 4])

t > 0

t2 = torch.tensor([[12, 16, 14], [13, 17, 13], [14, 18, 12]])
t2 * 2

"""#### <font color = 'pickle'>**Broadcasting a vector to matrix**</font>"""

t2 = torch.tensor([[12, 16, 14], [13, 17, 13], [14, 18, 12]])
t1 = torch.tensor([1, 2, 3])
print(t2.shape)
print(t1.shape)
print(t1 + t2)

"""### <font color = 'pickle'>**1) Understanding how broadcasting works**

* The following image describes how a tensor of 2 dimensional tensor will be added to a 1 dimensional tensor
<img src="https://drive.google.com/uc?export=view&id=1QG2GO1owGpyXbcugJFVFGb4o_buV4s3j" width="500"/>
"""

# create tensor
t2 = torch.tensor([[12, 16, 14], [13, 17, 13], [14, 18, 12]])
t1 = torch.tensor([1, 2, 3])
print(t2.shape)
print(t1.shape)

print(t1.storage())

t1_mod = t1.expand_as(t2)
t1_mod.shape

"""Although it appears as though we are copying the rows, we are actually not duplicating them."""

print(t1_mod.storage())

t1_mod

print(t1_mod + t2)

# we can check that it gives us the same result if we simply add t1 and t2
# so broadcasting is an efficient way of performing operations on tensors of unequal sizes
print(t1 + t2)

"""### <font color = 'pickle'>**2) Rules for Broadcasting**</font>
Broadcasting can only happen if the two tensors are broadcastable. Conditions for broadcasting:

- Each tensor has at least one dimension.

- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

Examples:

1. `t1(5, 8, 10) t2(5, 8, 10)`
Same size -> Broadcasting possible.
2. `t1((0,)) t2(5, 8, 10)`
t1 doesn't have atleast one dimension -> Broadcasting not possible.
3. `t1(5, 8, 10, 1) t2(8, 1, 1)` Broadcasting possible. Reasons:
  - 1st trailing position : both have size 1
  - 2nd trailing position : t2 has size 1
  - 3rd trailing position : both have size 8
  - 4th training position: t2 size doesn't exist but t2 has atleast 1 dimension.
"""

1, 8,1,1

# Broadcasting
t1 = torch.empty(5, 8, 10, 1)
t2 = torch.empty(  8, 1, 1,)
(t1 + t2).size()

"""The dimensions after broadcasting will be:

- If the number of dimensions are
 not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.

- Then, for each dimension size, the resulting dimension size is the max of the sizes along that dimension.
"""

# Another example for broadcasting
t1 = torch.empty(1)
t2 = torch.empty(3, 1, 7)
(t1 + t2).size()

# Example where broadcasting is not possible
t1 = torch.empty(5, 8, 10, 1)
t2 = torch.empty  ( 3, 1, 1)
(t1 + t2).size()

"""Here, at third trailing position sizes are not equal and none of them is 1, thus broadcasting is not possible.

## <font color = 'pickle'>**Reduction**

We can calculate the sum of all elemnets of a vector or a matrix of any shape. This can be done using the ***sum*** function.
"""

# Creating a vector
x = torch.arange(5)
print(x)

# This will do summation of all the elements of the vector : 0 + 1 + 2 + 3 + 4 = 10
print(x.sum())

# Creating a matrix
X = torch.arange(0, 10).reshape(2, 5)
X = X.to(torch.float32)
print(X)

# This will do summation of all the elements of the matrix
print(X.sum())
# This will takle the mean of all teh elements
print(X.mean())

"""We can also calculate the mean or average of all elements in a vector or a matrix by dividing the sum of elements by no. of elements.

By default, invoking the sum/mean finction on a tensor will give us a scaler (reduces the tensor along all its axes)

We can also calculate sum, along the rows or columns by specifying the value of parameter "axis".

axis = 0 will calculate sum along the rows while axis = 1 will calculate sum along the columns.
"""

# Creating a matrix A
A = torch.arange(0, 15, dtype = float).reshape(5, 3)
A

# Sum of elements along axis = 0
# row sum for each column
# Since we are taking sum along axis = 0, the input tensor reduces along axis 0
# The shape of A was ([5,3])
# After invoking sum along axis = 0, the shape reduces to ([3])
print(f'Shape before rediction{A.shape}')
A.sum(axis = 0), A.sum(axis=0).shape

# Sum of elements along axis =  1
# column sum for each row
# Since we are taking sum along axis = 1, the input tensor reduces along axis 1
# The shape of A was ([5,3])
# After invoking sum along axis = 1, the shape redices to ([5])
A.sum(axis = 1), A.sum(axis = 1).shape

# Rules of broadcasting:  A and A.sum(axis=1) are not broadcastable
print(A/A.sum(axis=1))

"""## <font color = 'pickle'>**Non-Reduction Sum**

As seem in above examples, invoking sum() or mean() will reduce number of dimensions. We can keep number of axis unchanged by passing argument keepdims = True.
"""

# When we pass argument keepdim=True, the shape will now be ([5,1]. The output has 2-dimensions
# if we do not pass the argument keepdim=True, the shape will be ([5]). The output has one-dimension
sum_A_0 = A.sum(axis=1, keepdim=True)
print(sum_A_0.shape)
print(sum_A_0)

# Let us now try operation : A/(sum(A, axis=0))
print(A/A.sum(axis=1, keepdim=True))

A

A = torch.randint(0, 10 , (2,3,4))

A

A.shape

B = A.sum(axis = 1, keepdim=True)

B.shape

# Cumulative sum of elements along rows
A.cumsum(axis = 0)

# Cumulative sum of elements along columns
A.cumsum(axis = 1)

"""## <font color = 'pickle'>**Accessing elements of a Tensor**

We can access individual elements of a Tensor using **index values**. Indexing always **starts from 0**.

For example if the tensor is: `[10, 12, 31, 34]`

Index of 10 is 0, index of 12 is 1 and so on.
"""

t1 = torch.tensor([[1, 2, 5], [7, 8, 9]])

# Printing all elements
print(t1)

# Get the first row
print(t1[0])

# Get the first element of the second row
print(t1[1][0])

# Get the first element of the second row
t1[1, 0]

"""## <font color = 'pickle'>**Accessing a Sub-Tensor**"""

t1 = torch.tensor([1, 5, 9, 13, 21, 45, 67, 34])

# Specify [from: to : step)
# by default step is 1
# here from is inclusive but to is not

# Get a subarray [9, 13, 21, 45]
# index 2 (i.e 9) is inclusive but index 6 (i.e. 67) is not and step size is 1
print(t1[2:6])

# Get a subarray [9, 21]
print(t1[2:6:2])  # step size is 2

# subarrays created using slicing and indexing do not create a copy,
# modifying the subarray modifies the original tensor as well

t1 = torch.tensor([1, 5, 9, 13, 21, 45, 67, 34])
t2 = t1[2:6]
print("array and subarray before modifying subarray")
print(t1)
print(t2)

# modify subarray

t2[0] = 100
print("\narray and subarray after modifying subarray")
print(t1)
print(t2)

"""<font color = 'indianred'> **As we can see above, modifying subarray, modifes the original array as well**"""

# use clone() method if yu specifically want to create a copy

t1 = torch.tensor([1, 5, 9, 13, 21, 45, 67, 34])
t2 = t1[2:6].clone()
print("array and subarray before modifying subarray")
print(t1)
print(t2)

# modify subarray

t2[0] = 100
print("\narray and subarray after modifying subarray")
print(t1)
print(t2)

"""<font color = 'indianred'> **SInce we created  a copy, modifying subarray, does not modify the original array as well**"""

t1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
t1

# get the sub array [[6,7], [10,11]]
t1[1:3, 1:3]

"""## <font color = 'pickle'>**Operations on tensors of same size**
We can call element-wise operations on any two tensors of the same shape.
"""

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y  # The ** operator is exponentiation

"""## <font color = 'pickle'>**Changing the shape of tensors**

### <font color = 'pickle'>**1) Reshape**

If we want to change the shape of our tensor, without affecting the elements present, we can use the ***reshape*** function.
"""

# Initializing a tensor with 10 elements from 0 to 9
t = torch.arange(10)
print(t)

# Changing the shape of tensor t from 1x10 to 2x5
tr = t.reshape(2, 5)
print(tr)

"""If we have to specify just 1 dimension in reshape function and want the function to calculate the second dimension itself, we can write `-1` in place of second dimension.

For 2 rows, we will write `reshape(2,-1)`

For 5 columns, we will write `reshape(-1,5)`.
"""

# Changing the shape of tensor t from 1 row to 2 rows
tr1 = t.reshape(2, -1)
print(tr1)

# Changing the shape of tensor t from 10 columns to 5 columns
tr2 = t.reshape(-1, 5)
print(tr2)

"""### <font color = 'pickle'>**2) View**

We can allow a tensor to be a view of an existing tensor. It performs the same operation as reshape. The only difference is that View will not create a copy and will allow us to perform fast and memory efficient computations whereas reshape may or may not share the same memory. There's a good discussion of the differences [here](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch).

Line from above link " *Another difference is that reshape() can operate on both contiguous and non-contiguous tensor while view() can only operate on contiguous tensor* "

[Definition of contiguous](https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays/26999092#26999092)

"""

# Initializing a tensor with 10 elements from 0 to 9
t = torch.arange(10)
print(t,'shape:', t.shape, sep='\n', end = '\n\n')
# Changing the shape of tensor t from 1x10 to 2x5
t = t.view(2, 5)
print(t,'shape:', t.shape, sep='\n')

"""Views can reflect changes from the base tensor."""

t = torch.arange(10)

# Create a view of tensor t
tr = t.view(2, 5)

# Before change in base tensor
print(f"before changing the base tensor\n{tr}")

# Modifying element of base tensor
t[0] = 67

# After change in base tensor
print(f"\nafter changing the base tensor\n {tr}")

# we can use -1 with view as well.
t = torch.rand((4, 5))
t1 = t.view(2, -1)
print(t1.shape)

# we can also flatten the tensor (convert the tensor to one dimensional tensor) by using view(-1)
# this gives the same result as method flatten()
t = torch.rand((4, 5, 3))
t2 = t.view(-1)
t3 = t.flatten()
print(t2.shape)
print(t3.shape)

"""### <font color = 'pickle'> **3) Adding and removing dimensions of size 1**
- Insert a dimension of size 1 at a specific location (location specified by dim) using `torch.unsqueeze(dim)`
- Remove a dimension of size 1 at a specific location (location specified by dim) using `torch.squeeze(dim)`
- Remove all dimensions of size 1 using `torch.squeeze()`
- Insert dimenion of size 1 using `None `keyword
- Remove dimenion of size 1 using `0 `keyword
"""

# Initialize an tensor
t1 = torch.ones(2, 2)
print(t1)
t1.shape

# add dimension of size 1 in the beginning using unsqueeze method and argument dim = 0
t1 = t1.unsqueeze(dim=0)
print(t1)
t1.shape

# add dimesnion of size 1 at the end usin unsqueeze method and dim = 3
t1 = t1.unsqueeze(dim=3)
print(t1)
print(t1.shape)

# We can add new dimesnion at any place
t1 = torch.arange(20).view(2, 2, 5)
print(t1.shape)
t1 = t1.unsqueeze(dim=1)
print(t1.shape)

# we can also use None keyword to add dimension of size 1 at multiple locations
t1 = t1[:, :, :, None, :, None]
print(t1.shape)

# Remove a dimension of size 1 at a specific location using torch.squeeze(dim)
t1 = t1.squeeze(dim=1)
print(t1.shape)

# Remove a dimension of size 1 at a specific location using 0 keyword
t1 = t1[:, :, 0]
print(t1.shape)

# Removing all dimensions of size 1 using torch.squeeze()
t1 = t1.squeeze()
print(t1.shape)

"""### <font color = 'pickle'>**4) Adopting shape of other tensors**
We can use view_as(input) to adopt shape of other tensors
"""

a = torch.arange(10).view(2, 5)
# create a tensor b filled with ones (10 elements) and has same shape as b
b = torch.ones(10).view_as(a)
print(a.shape)
print(b.shape)

"""### <font color = 'pickle'>**5) Permute**

Permute function rearranges the original tensor according to the desired ordering and returns a new multidimensional rotated tensor.

Let us consider an example:

If the size of a tensor is (2, 3, 4),

- First size is 2
- Second size is 3
- Third size is 4

Now, in case of permute we will just change the ordering of the sizes. Thus if we write permute(0, 2, 1) the new tensor will have:

- First size is 2 (1st size of previous)
- Second size is 4 (3rd size of previous)
- Third size is 3 (2nd size of previous)

Pytorch's function permute() only permutes or in other words shuffles the order of the axes of a tensor whereas view() reshapes the tensor by reducing/expanding the size of each dimension.

"""

# Initilaize a tensor and print it's size and elements
torch.manual_seed(0)
t1 = torch.randint(0, 10, size=(2, 4))
print(t1.size())
print(t1)

t1.storage()

t1.is_contiguous()

t1.stride()

# Permute the tensor and print it's size and elements
t1_p = t1.permute(1, 0)
print(t1_p.size())
print()
print(t1_p)

t1_p.storage()

t1_p.stride()

t1_p.is_contiguous()

t1_p.view(2, 4)

t1_p.reshape(2, 4)

# Initilaize a tensor and print it's size and elements
torch.manual_seed(0)
t2 = torch.rand(2, 3, 4)
print(t2.size())
print(f"\n{t2}")

# Permute the tensor and print it's size and elements - use permute (0, 2, 1)
t2_p = t2.permute(0, 2, 1)
print(t2_p.size())
print(f"\n{t2_p}")

# difference between permute and view
x = torch.arange(3 * 2).view(2, 3)
print(x)

# create a view (3, 2)
print(x.view(3, 2))

# permute axis(1, 0)
print(x.permute(1, 0))

"""Question and answer taken from following reference: <br>
https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916

- (1) If I have a feature size of BxCxHxW, I want to reshape it to BxCxHW. Which one is a good option?
- (2) If I have a feature size of BxCxHxW, I want to change it to BxCxWxH . Which one is a good option?
- (3) If I have a feature size of BxCxH, I want to change it to BxCxHx1 . Which one is a good option?

Solution:
- permute changes the order of dimensions aka axes, so 2 would be a use case. Transpose is a special case of permute, use it with 2d tensors.
- view can combine and split axes, so 1 and 3 can use view,
- note that view can fail for noncontiguous layouts (e.g. crop a picture using indexing), in these cases reshape will do the right thing,
- for adding dimensions of size 1 (case 3), there also are unsqueeze and indexing with None.

"""



"""## <font color = 'pickle'>**Concatenating Tensors**

We can use `torch.cat((tensors_to_concatenate), dim)` to concatenate tensors.

The tensors must have the same shape (except in the concatenating dimension).
"""

# we can use torch
x1 = torch.randint(low=0, high=10, size=(2, 5))
x2 = torch.ones(4, 5)
x3 = torch.zeros(2, 3)

# The tensors must have the same shape (except in the concatenating dimension)
# x1 and x2 have the same shape except for dim = 0, hence we can conactenate these along dim = 0
# x1 and x3 have the same shape except for dim = 1, hence we can conactenate these along dim = 1
# we cannot concatenate x2 and x3 along any dimension

x1_x2 = torch.cat((x1, x2), dim=0)
x1_x3 = torch.cat((x1, x3), dim=1)
print(f"shape of x1_x2 is {x1_x2.shape}")
print(f"shape of x2_x3 is {x1_x3.shape}")
print(f"\nx1_x2\n:{x1_x2}")
print(f"\nx1_x3\n:{x1_x3}")

"""## <font color = 'pickle'>**Some commonly used Tensors**

###<font color = 'pickle'>**1) Tensor containing all zeros/ all ones/ or any value**
"""

# Tensor containing all zeros, size = 10
z1 = torch.zeros(5)

# Tensor containing all zeros, size = 2 X 2 X 3
z2 = torch.zeros(2, 2, 3)

print(z1)
print(z2)

# Tensor containing all ones, size = 7
z1 = torch.ones(7)

# Tensor containing all ones, size = 1 X 2 X 3
z2 = torch.ones(1, 2, 3)

print(z1)
print(z2)

# We can also use torch.full(size, fill_value) to create a tensor filled with any value
# Tensor containing all fives, size = 1 X 2 X 3

z3 = torch.full(size=(2, 2, 3), fill_value=5)
print(z3)

"""### <font color = 'pickle'>**2) Tensor with elements in a particular range**
Suppose we need a tensor with values `1, 2, 3, 4.....n. `

We can simply specify the range and tensor will automatically get filled with these values.
"""

# Creating a tensor with integers from 1 to 5 with space 1: [1, 2, 3, 4, 5]
# syntax arange(start, end, step) - create tensor with values in the interval [start, end).
# start is inclusive , end is not i.e. start <= values < end
tr1 = torch.arange(1, 6)
print(tr1)

# Creating a tensor with integers from 0 to 10 with space 2 using "step" parameter: [0, 2, 4, 6, 8, 10]
tr2 = torch.arange(0, 11, step=2)
print(tr2)

"""We can also use `torch.linspace()` to generate evenly spaced values between two numbers"""

# Generate 10 evenly spaced values between 0 and 1 (both inclusive)
t1 = torch.linspace(0, 1, 10)
print(t1)

"""###<font color = 'pickle'>**3) Tensor with elements from probability distribution**

We can use the randn function to get elements from standard normal probabilty distribution i.e. normal dustribution with mean = 0 and variance = 1. If we want to select elements from normal ditsribution with different mean and variance then we should use torch.normal
"""

# for reproducabilty so that we get same results everytime we run this cell
torch.manual_seed(42)

# Sample 500,000 values from standard normal distribution (mean = 0 , variance = 1)
t1 = torch.randn(500000)

# Sample 500,000 values from normal distribution (mean = 5 , std = 2)
t2 = torch.normal(mean=5, std=2, size=(500000,))

print("Mean and std of tensor using torch.randn")
print(torch.mean(t1))
print(torch.std(t1))

print("\nMean and std of tensor using torch.normal")
print(torch.mean(t2))
print(torch.std(t2))

# for reproducabilty so that we get same results everytime we run this cell
torch.manual_seed(0)

# we sampled 10 values from standard noemal distribution. (5, 2) is the shape.
t1 = torch.randn(5, 2)
t1

"""We can also sample from other distributions like torch.rand, torch.randint etc.

### <font color = 'pickle'>**4) Empty Tensor**
We can create uninitialized  tensors using torch.empty.
"""

# create empty tensor of shape (2, 4)
empty_tensor = torch.empty(2, 4)
empty_tensor

"""### <font color = 'pickle'>**5) Commonly used tensors based on shape of other tensors**

We can also use `torch.zeros_like(input)`, `torch.ones_like(input)`, `torch.full_like(input)` and `torch.empty_like(input)` to create tensors based on the shape of other tensors
"""

input_tensor = torch.arange(6).view(2, 3)
input_tensor.shape

print(torch.ones_like(input_tensor))
print(torch.zeros_like(input_tensor))
print(torch.full_like(input_tensor, 5))
print(torch.empty_like(input_tensor))

"""###<font color = 'pickle'> **6) Identity Matrix**

Identity matrix is a matrix which has 1's along the diagnal and zeros everywhere else.
"""

# Identity matrix of size 3
id_matrix = torch.eye(3)
print(id_matrix)

# Identity matrix of size 5
id_matrix = torch.eye(5)
print(id_matrix)



"""## <font color = 'pickle'>**Masks using binary tensors**"""

# create a tensor which has probailities of events
prob = torch.tensor([0.7, 0.4, 0.6, 0.2, 0.8, 0.1])

# Binary tensors
print(prob > 0.5)
print(prob <= 0.5)

# create output tensor where output = 1 if prob >0.5 and 0 otherwise
# craete an empty output tensor of same shape as prob
output = torch.empty_like(prob)

# update output tensor using the binary mask
output[prob > 0.5] = 1
output[prob <= 0.5] = 0
print(output)

