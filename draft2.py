v = 0
s = 0
a = 1
map = """
..........
..........
..........
..........
..........
..........
..........
..........
..^.......
xxxxxxxxxx
"""
maplist = map.split()
map_width = len(maplist[0]) + 1
print map_width
start_pos = (1,1)

command_list = ['r','ur','ur','ur']
def look(map,pos):
    x,y = pos
    m = list(map)
    if m[x+(map_width*(y+1))-map_width] == "x":
        ground = 1
    else:
        ground = 0
    return ground
    
def move(command,pos,map):
    x,y = pos
    ground = look(map,pos)
    if command == 'r':
        if ground == 0:
            pos = (x+1,y+1)
        else:
            pos = (x+1,y)
    if command == 'ur':
        pos = (x+1,y-1)
    return pos
        
def show_game(map,pos):
    x,y = pos
    m = list(map)
    m[x+(map_width*y)-map_width] = "#"
    m = "".join(m)
    print m

def play_game(map,start_pos,command_list):
    pos = start_pos
    show_game(map,pos)
    for command in command_list:
        pos = move(command,pos,map)
        show_game(map,pos)
        
    
play_game(map,start_pos,command_list)