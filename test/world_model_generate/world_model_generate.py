import json

objects = []


def new_object(position, clazz, size, orientation):
    return {'position': position, 'class': clazz, 'orientation': orientation, 'size': size}


def new_wall(position1, position2, height):
    return {'position1': position1, 'position2': position2, 'height': height}


table = ([0, 0, 0], 1, 2, [0])

chair1 = ([2, 0, 2], 0, 1, [1.57])

chair2 = ([-2, 0, -2], 0, 1, [-1.57])

wall1 = ([-5, -5], [5, -5], 4)
wall2 = ([-5, -5], [-5, 5], 4)
wall3 = ([5, 5], [5, -5], 4)
wall4 = ([5, 5], [-5, 5], 4)

objects = []
walls = []

objects.append(new_object(*table))
objects.append(new_object(*chair1))
objects.append(new_object(*chair2))

walls.append(new_wall(*wall1))
walls.append(new_wall(*wall2))
walls.append(new_wall(*wall3))
walls.append(new_wall(*wall4))

print(objects)
print(walls)

fp_obj = open('../../data/temp_files/results/object_data.txt', 'w')
fp_wall = open('../../data/temp_files/results/wall_data.txt', 'w')

json.dump(objects, fp_obj)
json.dump(walls, fp_wall)


