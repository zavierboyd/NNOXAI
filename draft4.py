import numpy as np
from math import sqrt
vx_max = 1
vy_max = 3
vx_min = -1
vy_min = -2
vx = 0
vy = 0
sx = 0
sy = 0
ay = 1
alive = True
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
x....^^......x....x
x....^^.^...x.....x
xxxxxxxxxxxxxxxxxxx
"""






maplist = map.split()
map_width = len(maplist[0]) + 1
print map_width
start_pos = (2,1)

command_list = ['','','','','','', 'r','r','','ur','r','r','r','r','ur','ur','ur','ur','r','r','r','ur','r','ur','r','ur','r','r','r','l','l','l','l','l','l','l','l','l','u','u','u','u','u','u','u','u']

def look(map,pos,dx=0,dy=0):
    x,y = pos
    print pos
    m = list(map)
    return m[(x+dx)+(map_width*(y+dy))-map_width]
    
def tileaction(map,pos,dx=0,dy=0):
    global alive
    tile = look(map,pos,dx=dx,dy=dy)
    print tile
    if tile == spike and dx == 0 and dy == 0:
        print "You Died!"
        alive = False
    elif tile == finish and dx == 0 and dy == 0 and alive:
        print "You Win!"
    else:
        return tile

def calcvelocity(map,pos,command):
    global vy
    global vx
    print command
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
        
    elif command == "u" and look(map,pos,dy=1) == ground:
        vy -= 3
        if vy < vy_min:
            vy = vy_min
    
    elif command == "ul" and look(map,pos,dy=1) == ground:
        vy -= 3
        if vy < vy_min:
            vy = vy_min
        vx -= 1
        if vx < vx_min:
            vx = vx_min
        
    elif command == 'l':
        vx -= 1
        if vx < vx_min:
            vx = vx_min
        
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
        for i in range(abs(vy)):
            if vy > 0: dy = 1
            else: dy = -1
            print dy
            hit = False
            if xmove:
                print 'check'
                if tileaction(map,(x,y),dx=dx,dy=dy) == ground or tileaction(map,(x,y),dy=dy) == ground:
                    vy = 0
                    hit = True
                else:
                    y += dy
                    tileaction(map,(x,y))
                    tileaction(map,(x-dx,y))
            else:
                print 'check'

                if tileaction(map,(x,y),dy=dy) == ground:
                    vy = 0
                    hit = True
                else:
                    print 'move',y,dy
                    y += dy
                    print y
                    tileaction(map,(x,y))
            print hit
            if hit:
                break
    
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

print alive,'e'