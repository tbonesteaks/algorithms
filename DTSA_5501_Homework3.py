#!/usr/bin/env python
# coding: utf-8

# # Assignment 3 
# 
# ## Problem 1: Design a Correct Partition Algorithm
# 
# You are given code below for an incorrect partition algorithm that fails to partition arrays wrongly or cause out of bounds access in arrays.  The comments include the invariants the algorithm wishes to maintain and will help you debug.
# 
# Your goal is to write test cases that demonstrate that the partitioning will fail in various ways.
# 

# In[1]:


def swap(a, i, j):
    assert 0 <= i < len(a), f'accessing index {i} beyond end of array {len(a)}'
    assert 0 <= j < len(a), f'accessing index {j} beyond end of array {len(a)}'
    a[i], a[j] = a[j], a[i]

def tryPartition(a):
    # implementation of Lomuto partitioning algorithm
    n = len(a)
    pivot = a[n-1] # choose last element as the pivot.
    i,j = 0,0 # initialize i and j both to be 0
    for j in range(n-1): # j = 0 to n-2 (inclusive)
        # Invariant: a[0] .. a[i] are <= pivot
        #            a[i+1]...a[j-1] are > pivot
        if a[j] <= pivot: 
            swap(a, i+1, j)
            i = i + 1
    swap(a, i+1, n-1) # place pivot in its correct place.
    return i+1 # return the index where we placed the pivot


# First write a function that will return True if an array is correctly partitioned at index `k`. I.e, all elements at indices `< k` are all `<= a[k]` and all elements indices `> k` are all `> a[k]`

# In[16]:


def testIfPartitioned(a, k):
    # TODO : test if all elements at indices < k are all <= a[k]
    #         and all elements at indices > k are all > a[k]
    # return TRUE if the array is correctly partitioned around a[k] and return FALSE otherwise
    assert 0 <= k < len(a)
    #print(a[k])
    for i in range(0,k-1):
            if a[i] <= a[k]:
                pass
            else:
                return False
    for i in range(k+1,len(a)):
            if a[i] > a[k]:
                pass
            else:
                return False
    return True
    


# In[17]:


assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 14, 10, 23],5) == True, ' Test # 1 failed.'
assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 14, 11, 23],4) == False, ' Test # 2 failed.'
assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 14, 23, 21],0) == True, ' Test # 3 failed.'
assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 14, 22, 23],9) == True, ' Test # 4 failed.'
assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 14, 8, 23],5) == False, ' Test # 5 failed.'
assert testIfPartitioned([-1, 5, 2, 3, 4, 8, 9, 13, 9, -11],5) == False, ' Test # 6 failed.'
assert testIfPartitioned([4, 4, 4, 4, 4, 8, 9, 13, 9, 11],4) == True, ' Test # 7 failed.'
print('Passed all tests (10 points)')


# In[19]:


# Write an array called a1 that will be incorrectly partitioned by the tryPartition algorithm above
# Your input when run on tryPartition algorithm should raise an out of bounds array access exception
# in the swap function or fail to partition correctly. 

## Define an array a1 below of length > 0 that will be incorrectly partitioned by tryPartition algorithm.
## We will test whether your solution works in the subsequent cells.

a1 = [0,7,7,7,7,7,7,7,7,7]
assert( len(a1) > 0)

# Write an array called a2 that will be incorrectly partitioned by the tryPartition algorithm above
# Your input when run on tryPartition algorithm should raise an out of bounds array access exception
# in the swap function or fail to partition correctly. 
# a2 must be different from a1
 
a2 = [3,3,3,3,3,3]
assert( len(a2) > 0)
assert (a1 != a2)


# Write an array called a3 that will be incorrectly partitioned by the tryPartition algorithm above
# Your input when run on tryPartition algorithm should raise an out of bounds array access exception
# in the swap function or fail to partition correctly. 
# a3 must be different from a1, a2

a3 = [12,12,22,2,7,2,22,12]
assert( len(a3) > 0)
assert (a3 != a2)
assert (a3 != a1)

def dummyFunction():
    pass


# In[21]:


try:
    j1 = tryPartition(a1)
    assert not testIfPartitioned(a1, j1)
    print('Partitioning was unsuccessful - this is what you were asked to break the code')
except Exception as e: 
    print(f'Assertion failed {e} - this is fine since you were asked to break the code.')
    
try:
    j2 = tryPartition(a2)
    assert not testIfPartitioned(a2, j2)
except Exception as e: 
    print(f'Assertion failed {e} - this is fine since you were asked to break the code.')
    

try:
    j3 = tryPartition(a3)
    assert not testIfPartitioned(a3, j3)
except Exception as e: 
    print(f'Assertion failed {e} - this is fine since you were asked to break the code.')
    
dummyFunction()

print('Passed 5 points!')


# In[44]:


# Troubleshoot the function and test on a1, a2, and a3
# First pass we'll do this the hard way. 
def tryPartition2(a):
    # implementation of Lomuto partitioning algorithm
    n = len(a)
    #print(n)
    pivot = a[n-1] # choose last element as the pivot.
    i,j = 0,0 # initialize i and j both to be 0
    for j in range(n-2): # j = 0 to n-2 (inclusive)
        # Invariant: a[0] .. a[i] are <= pivot
        #            a[i+1]...a[j-1] are > pivot
        if a[j] <= pivot: 
            swap(a, i, j)
            i += 1
            
    print(a, "Pivot Index:", i, " Pivot Value:", a[n-1] , "\r")
    swap(a, i, n-1) # place pivot in its correct place.
    return i # return the index where we placed the pivot

# Second pass we'll do this the easy way. 
# Only thing that changes is the initialization of i = -1
def tryPartition3(a):
    # implementation of Lomuto partitioning algorithm
    n = len(a)
    pivot = a[n-1] # choose last element as the pivot.
    i,j = -1,0 # initialize i and j both to be 0
    for j in range(n-1): # j = 0 to n-2 (inclusive)
        # Invariant: a[0] .. a[i] are <= pivot
        #            a[i+1]...a[j-1] are > pivot
        if a[j] <= pivot: 
            swap(a, i+1, j)
            i = i + 1
    
    print(a, "Pivot Index:", i, " Pivot Value:", a[n-1] , "\r")
    swap(a, i+1, n-1) # place pivot in its correct place.
    return i+1 # return the index where we placed the pivot
print("Original Array: ",a1)
tryPartition2(a1)
print("\n Original Array: ",a2)
tryPartition2(a2)

print("\n Original Array: ",a3)
tryPartition3(a3)


# ### Debug the function
# 
# Point out where the bug is and what the fix is for the tryPartition function. Note that the answer below is not graded.
First pass is the hard way:

This line:

    swap(a, i+1, j)

Should be:
    
    swap(a, i, j)
    
Final swap should be i and n-1 (not i+1 and n-1)

Return i as index location, not i+1

Second pass is the easy way:

Just initialize i as -1 to correct for the 0 index in python.
# ## Problem 2. Rapid Sorting of Arrays with Bounded Number of Elements.
# 
# Thus far, we have presented sorting algorithms that are comparison-based. Ie., they make no assumptions about the elements in the array just that we have a `<=` comparison operator. We now ask you to develop a rapid sorting algorithm for an array of size $n$ when it is given to you that all elements in the array are between $1, \ldots, k$ for a given $k$. Eg., consider an array with n = 100000 elements wherein all elements are between 1,..., k = 100.
# 
# 
# Develop a sorting algorithm using partition that runs in $\Theta(n \times k)$ time for such arrays. __Hint__ You can choose your pivots in a simple manner each time. 
# 
# ### Part A
# 
# Describe your algorithm as pseudocode and argue why it runs in time $\Theta(n \times k)$. This part will not be graded but is intended for your own edification.

# for i in range k:
#     
#     pivot on the values, and increment i to determine index after each swap.
#     Have to run through the values (n length) in order (k times) but we are never touching right of index (i).
#     Therefore, 
#         average case: is far less than Θ(n*k) because as i creeps across, array n to compare is being reduced.  
#         worst case: (all elts are k, or 100 here), we run through them each time and i never increments.

# ## Part B 
# Complete the implementation of a function `boundedSort(a, k)` by completing the `simplePatition` function. Given an array `a` and a fixed `pivot` element, it should partition the array "in-place" so that all elements `<= pivot` are on one side of the array and elements `> pivot` on the other.  You should not create a new array in your code.

# In[47]:



def swap(a, i, j):
    assert 0 <= i < len(a), f'accessing index {i} beyond end of array {len(a)}'
    assert 0 <= j < len(a), f'accessing index {j} beyond end of array {len(a)}'
    a[i], a[j] = a[j], a[i]

def simplePartition(a, pivot):
    ## To do: partition the array a according to pivot.
    # Your array must be partitioned into two regions - <= pivot followed by elements > pivot
    ## If an element at the beginning of the array is already <= pivot in the beginning of the array, it should not
    ## be moved by the algorithm.
    n = len(a) 
    i,j = -1,0
    for j in range(n):
        if a[j] <= pivot:
            swap(a,i+1,j)
            i +=1   
def boundedSort(a, k):
    for j in range(1, k):
        simplePartition(a, j)


# In[48]:


a = [1, 3, 6, 1, 5, 4, 1, 1, 2, 3, 3, 1, 3, 5, 2, 2, 4]
print(a)
simplePartition(a, 1)
print(a)
assert(a[:5] == [1,1,1,1,1]), 'Simple partition test 1 failed'

simplePartition(a, 2)
print(a)
assert(a[:5] == [1,1,1,1,1]), 'Simple partition test 2(A) failed'
assert(a[5:8] == [2,2,2]), 'Simple Partition test 2(B) failed'


simplePartition(a, 3)
print(a)
assert(a[:5] == [1,1,1,1,1]), 'Simple partition test 3(A) failed'
assert(a[5:8] == [2,2,2]), 'Simple Partition test 3(B) failed'
assert(a[8:12] == [3,3,3,3]), 'Simple Partition test 3(C) failed'

simplePartition(a, 4)
print(a)
assert(a[:5] == [1,1,1,1,1]), 'Simple partition test 4(A) failed'
assert(a[5:8] == [2,2,2]), 'Simple Partition test 4(B) failed'
assert(a[8:12] == [3,3,3,3]), 'Simple Partition test 4(C) failed'
assert(a[12:14]==[4,4]), 'Simple Partition test 4(D) failed'

simplePartition(a, 5)
print(a)
assert(a == [1]*5+[2]*3+[3]*4+[4]*2+[5]*2+[6]), 'Simple Parition test 5 failed'

print('Passed all tests : 10 points!')


# ## Problem 3: Design a Universal Family Hash Function

# Suppose we are interested in hashing $n$ bit keys into $m$ bit hash values to hash into a table of size
# $2^m$. We view our key  as a bit vector of $n$ bits in binary. Eg., for $n = 4$, the key $14 = \left(\begin{array}{c} 1\\ 1\\ 1\\ 0 \end{array} \right)$.
# 
# The hash family is defined by random boolean matrices $H$ with $m$ rows and $n$ columns. To compute the hash function, we perform a matrix multiplication. Eg., with $m = 3$ and $n= 4$, we can have a matrix $H$ such as
# 
# $$ H = \left[ \begin{array}{cccc} 0 & 1 & 0 & 1 \\
# 1 & 0 & 0 & 0 \\
# 1 & 0 & 1 & 1 \\
# \end{array} \right]$$.
# 
# 
# The value of the hash function $H(14)$ is now obtained by multiplying
# 
# $$ \left[ \begin{array}{cccc} 0 & 1 & 0 & 1 \\
# 1 & 0 & 0 & 0 \\
# 1 & 0 & 1 & 1 \\
# \end{array} \right] \times \left( \begin{array}{c} 
# 1\\
# 1\\
# 1\\
# 0
# \end{array} \right) $$
# 
# The matrix multiplication is carried out using AND for multiplication and XOR instead of addition. For the example above, we compute the value of hash function as
# 
# $$\left( \begin{array}{c} 
#  0 \cdot 1 + 1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 \\
#  1 \cdot 1 + 0 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 \\
#  1 \cdot 1 + 0 \cdot 1 + 1 \cdot 1 + 1 \cdot 0 \\
#  \end{array} \right) = \left( \begin{array}{c} 1 \\ 1 \\ 0 \end{array} \right)$$
# 
# (A) For a given matrix $H$ and two  keys $x, y$ that differ only in their $i^{th}$ bits, provide a condition for
# $Hx = Hy$ holding. (**Hint** It may help to play with examples where you have two numbers $x, y$ that just differ at a particular bit position. Figure out which entries in the matrix are multiplied with these bits that differ).
# 

# YOUR ANSWER HERE

# 
# (B) Prove that the probability that two keys $x, y$ such that $x \not= y$ collide under the random choice of a matrix $x, y$ is at most $\frac{1}{2^m}$.
# 

# YOUR ANSWER HERE

# In[60]:


from random import random

def dot_product(lst_a, lst_b):
    and_list = [elt_a * elt_b for (elt_a, elt_b) in zip(lst_a, lst_b)]
    return 0 if sum(and_list)% 2 == 0 else 1

# encode a matrix as a list of lists with each row as a list.
# for instance, the example above is written as the matrix
# H = [[0,1,0,1],[1,0,0,0],[1,0,1,1]]
# encode column vectors simply as a list of elements.
# you can use the dot_product function provided to you.
def matrix_multiplication(H, lst):
    hasher = []
    for i in H:
        hasher.append(dot_product(i,lst))
    return hasher
    

# Generate a random m \times n matrix
# see the comment next to matrix_multiplication for how your matrix must be returned.
def return_random_hash_function(m, n):
    # return a random hash function wherein each entry is chosen as 1 with probability >= 1/2 and 0 with probability < 1/2
    rethash = []
    for i in range(m):
        fhash = []
        for i in range(n):
            if random() > .4:
                fhash.append(1)
            else:
                fhash.append(0)
        rethash.append(fhash)
    return rethash


# In[61]:


A1 = [[0,1,0,1],[1,0,0,0],[1,0,1,1]]
b1 = [1,1,1,0]
c1 = matrix_multiplication(A1, b1)
print('c1=', c1)
assert c1 == [1,1,0] , 'Test 1 failed'

A2 = [ [1,1],[0,1]]
b2 = [1,0]
c2 = matrix_multiplication(A2, b2)
print('c2=', c2)
assert c2 == [1, 0], 'Test 2 failed'

A3 = [ [1,1,1,0],[0,1,1,0]]
b3 =  [1, 0,0,1]
c3 = matrix_multiplication(A3, b3)
print('c3=', c3)
assert c3 == [1, 0], 'Test 3 failed'

H = return_random_hash_function(5,4)
print('H=', H)
assert len(H) == 5, 'Test 5 failed'
assert all(len(row) == 4 for row in H), 'Test 6 failed'
assert all(elt == 0 or elt == 1 for row in H for elt in row ),  'Test 7 failed'

H2 = return_random_hash_function(6,3)
print('H2=', H2)
assert len(H2) == 6, 'Test 8 failed'
assert all(len(row) == 3 for row in H2),  'Test 9 failed'
assert all(elt == 0 or elt == 1 for row in H2 for elt in row ), 'Test 10 failed'
print('Tests passed: 10 points!')


# ## Manually Graded Answers
# 
# ### Problem 1
# 
# The bug is in the initialization of i in the algorithm. It must be i =-1 rather than i = 0. Due to this, either the first element of the array is never considered during the partition or there could be an access to i+1 that is out of array bounds.
# 
# ### Problem 2 A 
# 
# ~~~
# for k = 1 to n
#    j = partition array a with k as pivot
# ~~~   
# The running time is $\Theta(n \times k)$.
# 
# ### Problem 3 A 
# Since  $x,y$ differe only in their $i^{th}$ bits, we can assume $x_i = 0$ and $y_i = 1$.
# Therefore, $H x  + H_i = Hy$ wherein, $+$ refers to entrywise XOR and $H_i$ is the $i^{th}$ column of $H$.
# Thus, $Hx = Hy$ if and only if $H_i$ has all zeros.  This happens with probability $\frac{1}{2^m}$.
# 
# ### Problem 3 B 
# Let us assume that $x$ and $y$ differ in $k$ out of $n$ positions. We know that $Hx = Hy$ if and only
# if $Hx + Hy = 0$ where $+$ is XOR and $0$ is the vector of all zeros. But $Hx + Hy = H (x + y)$ since AND distributes over XOR.
# 
# Whenever $x$ and $y$ agree in the $i^{th}$ entries, we have the $i^{th}$ entry of $(x+y)$ is zero.
# Therefore, $H(x+y)$ is just the XOR sum of $k$ columns of $H$ corresponding to positions where $x$ and $y$ differ. 
# 
# Thus, one of the columns must equal the sum of the remaining $k-1$ columns. Let us fix these $k-1$ columns as given and the last column as randomly chosen. The probability that each of the $m$ entries of the last column matches the sum of the first $k-1$ column is $\frac{1}{2^m}$.

# ## That's all folks
