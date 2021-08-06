#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:55:11 2021

@author: allan
"""

program="architecture dataflow of ind is begin o(0) <= (b(1) and a(1)) ; o(1) <= ((a(1) and (b(0) and a(1))) xor ((b(1) xor (a(0) and (((((b(0) and a(1)) and b(0)) and a(0)) xor (((a(0) xor a(1)) xor (a(1) and ((a(0) xor a(1)) xor b(0)))) and a(0))) and (((b(0) xor a(1)) xor (a(1) and (b(0) and a(0)))) and ((((b(1) and a(0)) xor a(1)) xor b(1)) and (((a(1) xor ((((a(0) xor b(0)) xor a(1)) xor a(0)) and b(1))) and (b(1) xor b(1))) and (b(0) and a(1)))))))) and ((b(1) xor (a(1) and ((b(1) xor (b(0) and a(1))) and b(0)))) and a(0)))) ; o(2) <= (((a(0) xor a(1)) xor (a(1) and (b(0) and ((a(1) and (b(0) and a(1))) xor ((b(1) xor (a(1) and ((b(1) xor (b(0) and b(1))) and b(0)))) and a(0)))))) and (((a(0) xor a(1)) xor (a(1) and (b(0) and a(1)))) and (b(0) and a(0)))) ; o(3) <= (((b(0) xor b(0)) xor ((((a(0) xor a(1)) xor (a(1) and b(1))) and b(1)) and (b(0) and a(1)))) and ((b(1) xor (a(1) and ((b(1) xor (b(0) and a(1))) and b(0)))) and a(0))) ; end dataflow;"

program_splitted = program.split(";")

o0 = program_splitted[0].split("<=")[1]
o1 = program_splitted[1].split("<=")[1]
o2 = program_splitted[2].split("<=")[1]
o3 = program_splitted[3].split("<=")[1]

o2 = o2.replace("(0)","[0]")
o2 = o2.replace("(1)","[1]")
o2 = o2.replace("(","( ")
o2 = o2.replace(")"," )")

 
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree

         
def longestPath(root):
     
    # If root is null means there
    # is no binary tree so
    # return a empty vector
    if (root == None):
        return []
 
    # Recursive call on root.right
    rightvect = longestPath(root.getRightChild())
 
    # Recursive call on root.left
    leftvect = longestPath(root.getLeftChild())
 
    # Compare the size of the two vectors
    # and insert current node accordingly
    if (len(leftvect) > len(rightvect)):
        leftvect.append(root.key)
    else:
        rightvect.append(root.key)
 
    # Return the appropriate vector
    if len(leftvect) > len(rightvect):
        return leftvect
 
    return rightvect

o4 = "( a[0] and ( ( a[1] and ( a[1] xor a[0] ) ) xor b[0] ) )"    

def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        #elif i not in ['+', '-', '*', '/', ')']:
        elif i not in ['and', 'xor', ')']:
            currentTree.setRootVal(str(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in ['and', 'xor']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError
    return eTree

#pt = buildParseTree("( ( 10 + 5 ) * 3 )")
pt = buildParseTree(o2)
print(longestPath(pt))


pt.postorder()  #defined and explained in the next section

t= buildParseTree(o4)



string = str(o0)

currentDepth = 0
maxDepth = 0
xor2 = 0
and2 = 0
for c in string:
    if c == '(':
        currentDepth += 1
    elif c == ')':
        currentDepth -= 1
    elif c == 'x':
        xor2 += 1
    elif c == 'a':
        and2 += 1

    maxDepth = max(maxDepth, currentDepth)
    
print(maxDepth)



# Python3 program to find the maximum depth of tree

# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
 
# Compute the "maxDepth" of a tree -- the number of nodes
# along the longest path from the root node down to the
# farthest leaf node
def maxDepth(node):
    if node is None:
        return 0 ;
 
    else :
 
        # Compute the depth of each subtree
        lDepth = maxDepth(node.left)
        rDepth = maxDepth(node.right)
 
        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1
 
 
# Driver program to test above function
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
 
 
print ("Height of tree is %d" %(maxDepth(root)))