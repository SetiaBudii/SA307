import numpy as np

def generate_directional_points(points, positive_point_count):
    """
    Generate point baru berdasarkan titik awal dan jumlah titik positif yang diinginkan

    Args:
        points (np.ndarray): Point awal yang berisi koordinat (x, y).
        positive_point_count (int): Jumlah titik positif yang diinginkan.

    Returns:
        new_prompt (np.ndarray): Array yang menggabungkan titik asli dan titik baru.
        input_label (np.ndarray): Array label (semua satu).
    """
    first_point = points[0]
    new_points = []

    if positive_point_count >= 1:
        new_points.append((first_point[0] - 5, first_point[1]))
    if positive_point_count >= 2:
        new_points.append((first_point[0] + 5, first_point[1]))
    if positive_point_count >= 3:
        new_points.append((first_point[0] - 10, first_point[1]))
    # Tambahkan pola lain jika diperlukan untuk positive_point_count > 3

    if new_points:
        new_points = np.array(new_points)
        new_prompt = np.concatenate([points, new_points], axis=0)
    else:
        new_prompt = points

    input_label = np.ones(len(new_prompt), dtype=int)
    return new_prompt, input_label

def get_left_point(point, shift):
  return [point[0] - shift, point[1]]

def get_right_point(point, shift):
  return [point[0] + shift, point[1]]

def get_horizontal_point(point, num, shift=5):
  new_points = []
  for i in range(1, num + 1):
      if len(new_points) >= num:
          break
      new_points.append(get_left_point(point, shift * i))
      if len(new_points) >= num:
          break
      new_points.append(get_right_point(point, shift * i))
  return new_points