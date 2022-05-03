import os
from registration import registrate
rooms = [
    'scene0011_00', 'scene0011_01', 'scene0015_00', 'scene0019_00',
    'scene0025_00', 'scene0025_01', 'scene0025_02'
]
for i in range(len(rooms)):
    registrate(os.path.join('./pcds/gt', rooms[i]))
