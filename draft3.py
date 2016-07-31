import numpy as np
from math import sqrt
vx_max = 2
vy_max = 3
vx_min = -2
vy_min = -3
vx = 0
vy = 0
sx = 0
sy = 0
ay = 1
player = "@"
ground = "x"
finish = "f"
space = "."
spike = "^"
map = """
x.................x
x.................x
x.................x
x.................x
x.................x
x.................x
x.................x
x.................x
x.................x
x.................x
x................fx
x..............xxxx
x.............x...x
x............x....x
x....^......x.....x
xxxxxxxxxxxxxxxxxxx
"""






maplist = map.split()
map_width = len(maplist[0]) + 1
print map_width
start_pos = (2,1)

command_list = ['','','','','','', 'r','r','','ur','r','r','r','r','ur','ur','ur','ur','r','r','r','ur','r','ur','r','ur','r','r','r']

def look(map,pos,dx=0,dy=0):
    x,y = pos
    print pos
    m = list(map)
    return m[(x+dx)+(map_width*(y+dy))-map_width]
    
def tileaction(map,pos,dx=0,dy=0):
    tile = look(map,pos,dx=dx,dy=dy)
    if tile == ground:
        return ground
    elif tile == spike and dx == 0 and dy == 0:
        print "You Died!"
        raise Exception("You Died!")
    elif tile == finish and dx == 0 and dy == 0:
        print "You Win!"
        raise Exception("You Won!")
    else:
        return space

def calcvelocity(map,pos,command):
    global vy
    global vx
    
    x,y = pos
    if command == 'r':
        vx += 1
        if vx > vx_max:
            vx = vx_max
            
    elif command == "ur" and look(map,pos,dy=1) == ground:
        vy -= 3
        if vy < vy_min:
            vy = vy_min
        vx += 1
        if vx > vx_max:
            vx = vx_max
        
    else:
        vx = 0

    if look(map,pos,dy=1) != ground:
        vy += ay
        if vy > vy_max:
            vy = vy_max
    
def move(map,pos):
    global vy
    global vx
    
    x,y = pos
    xmove = False
    if vx != 0:
        dx = abs(vx)/vx
        if tileaction(map,pos,dx=dx) == ground:
            vx = 0
            xmove = False
        else:
            x += dx
            tileaction(map,(x,y))
            xmove = True
    
    if vy != 0:
        dy = abs(vy)/vy
        if xmove:
            if tileaction(map,pos,dx=dx,dy=dy) == ground or tileaction(map,pos,dy=dy) == ground:
                vy = 0
            else:
                y += dy
                tileaction(map,(x,y))
                tileaction(map,(x-dx,y))
        else:
            if tileaction(map,pos,dy=dy) == ground:
                vy = 0
            else:
                y += dy
                tileaction(map,(x,y))
    
    pos = (x,y)
    tileaction(map,pos)
    return pos
    
            
            

    
def show_game(map,pos):
    x,y = pos
    m = list(map)
    m[x+(map_width*y)-map_width] = "@"
    m = "".join(m)
    print m

def play_game(map,start_pos,command_list):
    pos = start_pos
    show_game(map,pos)
    for command in command_list:
        calcvelocity(map,pos,command)
        pos = move(map,pos)
        show_game(map,pos)
        
    
play_game(map,start_pos,command_list)