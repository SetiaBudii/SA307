import numpy as np


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points

def swap_xy2(points):
    new_points = np.zeros((len(points),2))
    new_points[0, 0] = points[0, 0]
    new_points[0, 1] = points[0, 1]
    new_points[1:,0] = points[1:,1]
    new_points[1:,1] = points[1:,0]
    return new_points

def get_random_point(mask):
  indices = np.argwhere(mask==True)

  random_point = indices[np.random.choice(list(range(len(indices))))]
  random_point = [random_point[1], random_point[0]]
  return random_point
    
def get_multi_random_point(mask, points_nubmer):
    indices = np.argwhere(mask==True)

    random_point = indices[np.random.choice(list(range(len(indices))),points_nubmer,replace=False)]
    new_points = swap_xy(random_point)
    return new_points