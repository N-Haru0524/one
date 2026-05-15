import numpy as np
def spiral(num, step=0.0002):
    pts = []
    for i in range(num):
        theta = 2 * np.pi * i / num * (num / (num // 6 + 1))

        r = step * theta / (2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y])

    return np.array(pts)

def hex_ring_abs(num_classes, step=0.0002):
    """
    Absolute hexagonal ring layout.
    class 0 is center.
    Returns (num_classes, 2) array.
    """
    coords = [(0.0, 0.0)]
    directions = np.array([
        [ 1.0,  0.0],
        [ 0.5,  np.sqrt(3)/2],
        [-0.5,  np.sqrt(3)/2],
        [-1.0,  0.0],
        [-0.5, -np.sqrt(3)/2],
        [ 0.5, -np.sqrt(3)/2],
    ])
    ring = 1
    while len(coords) < num_classes:
        pos = ring * directions[4]

        for d in range(6):
            for _ in range(ring):
                if len(coords) >= num_classes:
                    break
                coords.append((pos * step).copy())
                pos = pos + directions[d]
        ring += 1
    return np.array(coords[:num_classes])
