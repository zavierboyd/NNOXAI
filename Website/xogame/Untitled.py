from __future__ import division
import random
import numpy as np
with open('urls.txt', 'r') as links:
    links = [line[:-1] for line in links.readlines()]

boards = [link[-11:-2] for link in links]
print boards
htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in boards]
print htmlboards[0]
print links
links.pop(0)
print links
urls = links

def transboard(board,side):
    oside = 'o' if side == 'x' else 'x'
    side = [1 if bit == side else 0 for bit in board]
    oside = [1 if bit == oside else 0 for bit in board]
    space = [1 if bit == '.' else 0 for bit in board]
    return side+oside+space

inputs = [transboard(list(board),'x') for board in boards]