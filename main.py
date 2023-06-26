import numpy as np
import matplotlib.pyplot as plt
from preprocessing import NoiseCancelPreprocessor, NoiseCancelQualityAssessor
from visuals import Visualizer


data = np.load(
    "data/data/AlN_ccurrent_50Ohm_2400V_30kHz_000283.npz"
)['data']

t = data[0]
current = data[2]

k = 20
preprocessor = NoiseCancelPreprocessor(
    data_x_values=t, data_y_values=current, k=k
)
preprocessor.preprocess("sliding_mean")
preprocessor.preprocess("limit", 0.001)
t_pr, current_pr = preprocessor.get_result()

assessor = NoiseCancelQualityAssessor(
    y_original=current, y_filtered=current_pr
)
print(assessor.get_snr_comparisons())

visualizer = Visualizer(
    data_x_values=t_pr,
    data_y_values=current_pr,
    x_label="time",
    y_label="I"
)
visualizer.draw("batches", 25)

