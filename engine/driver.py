from model import make_pred, best_n, load_relevant_data_subset, map_bn
from data import keypoints_from_camera
import numpy as np

while True:
    keypoints = keypoints_from_camera()
    data = load_relevant_data_subset('video_df.parquet')

    idxs = best_n(make_pred(data), 10)
    print(map_bn(idxs))

    again = input("Another one?: ")

    if again == "no":
        break

