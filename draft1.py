map = """
......
..^...
xxxxxx
"""
maplist = map.split()
map_width = len(maplist[0]) + 1
print map_width
start_pos = (1,2)

command_list = ['r','ur','r','r']

def move(command,pos):
    x,y = pos
    if command == 'r':
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
        pos = move(command,pos)
        show_game(map,pos)
        
    
play_game(map,start_pos,command_list)
