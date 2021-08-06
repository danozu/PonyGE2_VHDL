#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:47:30 2021

@author: allan
"""

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

def buildParseTree(fpexp, gates):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    
    aux = gates.copy()
    aux.append(')')
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        #elif i not in ['+', '-', '*', '/', ')']:
        elif i not in aux:#['and', 'xor', ')']:
            currentTree.setRootVal(str(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in gates:#['and', 'xor']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError
    return eTree

def count_gates_critical_path(phenotype,gates):
    n_outputs = phenotype.count("<=")
    phenotype_splitted = phenotype.split(";") #the first n_outputs positions contains the outputs
    output = []
    outputTree = []
    l_outputTree = []
    
    for i in range(n_outputs):
        output.append(phenotype_splitted[i].split("<=")[1])
        output[i] = output[i].replace("(0)","[0]")
        output[i] = output[i].replace("(1)","[1]")
        output[i] = output[i].replace("(","( ")
        output[i] = output[i].replace(")"," )")
        outputTree.append(buildParseTree(output[i],gates))
        l_outputTree.append(len(longestPath(outputTree[i])))
        
    l_criticalPath = max(l_outputTree)
    index_criticalPath = l_outputTree.index(l_criticalPath)

    n_gates_critical_path = l_criticalPath - 1 #because there is one node with a terminal

    return n_gates_critical_path