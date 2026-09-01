import numpy as np

def preprocess(frame, config):
    points = np.asarray(frame["points"], dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    points = points[np.isfinite(points).all(axis=1)]
    # Lightweight statistical clipping around the usable map extent.
    w, h = config.map_dimensions
    mask = (
        (points[:,0] >= -w/2) & (points[:,0] <= w/2) &
        (points[:,1] >= -h/2) & (points[:,1] <= h/2) &
        (points[:,2] >= -2) & (points[:,2] <= 8)
    )
    return points[mask]
