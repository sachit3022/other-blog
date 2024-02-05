+++
author = "Sachit gaudi"
title = "Tiktok online assesment"
date = "2024-01-15"
description = "Solving TikTok coding questions"
math = true
pygmentsUseClasses=true
pygmentsCodeFences=true
tableOfContents = true
+++



The Assesment comprises of solving 3 coding question, Medium to hard level leetcode questions. 
{{<toc>}}

### Maximise the maximum possible frequency of frequency counter
You are given a frequecy counter array <code> freq = [2,3,4] </code> the objective is to perform any one of the following operations to each of the element in the frequency array.
- You can add `a` quantity to the element
- Subtract `s` quantity from the element
- Multiply `m` quantity to the element
- leave the element unchanged

The objective is to maximise the highest frequency element, return the highest possible frequency after the following operations.

```Python
class MaxFrequnecy:
    def __init__(self,freq,a,s,m):
        self.freq = freq
        self.a,self.s,self.m = a,s,m
    def maximise(self):
        """Maximise the maximum possible frequency of frequency counter"""
        pass
```
:thinking: The idea is to perform all the possible operations on an element and form a set and then run the counter on all the sets this will give the maximum possible element and its frequency after performing the operations.


###  Splitwise: Minimum number of transactions to settle

The problem is to have 2 functions in a class one to take all the transactions in the format `(from,to,ammout)`. After adding the transactions once transactions, once the settle function is called, we need to calculate minimum number of transactions required to settle all dues.
```Python
class Splitwise:
    def __init__(self,N):
        """N is number of persons in a group."""
        self.N = N
        self.transactions = []
    def add_transactions(self,transaction):
        """transaction -> (from,to,ammout) 
        to person owes from person respective ammount."""
        pass
    def settle(self):
        """returns minimum number of transactions required to settle."""
        pass
```



:thinking: One observation we can make is minimum number of transactions is independent of transactions, rather it is dependent on the number of persons ($N$) and their outstanding balances $
 \begin{bmatrix}
    a & b & c
       \end{bmatrix}
 $ . We can even have a upper bound which is $N-1$, the easiest way is $\begin{bmatrix}
    0 & b+a & c
       \end{bmatrix}
 $ $ \rightarrow $ $\begin{bmatrix}
    0 & 0 & c+b+a
       \end{bmatrix}
 $ $\implies $ $\begin{bmatrix}
    0 & 0 & 0
       \end{bmatrix}
 $. We can improve this further by grouping into smaller subsets of persons who have outstanding balances sum to 0. Suppose, if you have person 0 and 3 sum to 0 then first settlement should me made between them. So you need to start with 2 pairs and so on till $N-2$ pairs to find settlements which is an NP-Complete problem, So time complexity is $O(2^N)$, but if the N is small we can have this but if the N is large we have the upperbound, which is a $O(N)$, solution

 


### Server allocation

You have $N$ servers from $1,2 \cdots N$. We have the process which have the `start_time`, `time_to_process`. The process will occupy the least possible server. We need to return the server number for all the process. If all the servers are busy we need to drop that process and return -1.
 
```Ballerina
For example,
N = 2
process = [(1,5),(2,1),(4,2),(5,10)]
output = [1,2,2,-1]
```

:thinking: This is a classic 2 heap problem, store the available servers on one heap and process end time on another heap. at current time if the end time of the process is less than the current time then pop it from the process heap and add the server to server heap. and for the current process if the server heap is not empty pop a server and add end endtime back to the process heap.
 

