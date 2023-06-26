import numpy.typing as ntp
import numpy as np
import scipy
from typing import Dict

from validator import Validator


class NoiseCancelQualityAssessor:
    """Takes in filtered and non-filtered signal and assesses
    noise-cancelling quality and amount of information lost
    """

    def __init__(
            self,
            y_original: ntp.NDArray,
            y_filtered: ntp.NDArray
    ) -> None:
        self.original = y_original
        self.filtered = y_filtered

    @staticmethod
    def __get_signal_statistics(y: ntp.NDArray):
        def autocorrelation(y: ntp.NDArray):
            y_centered = y - np.mean(y)
            return np.correlate(
                y_centered, y_centered, mode="full"
            ) / np.sum(y_centered ** 2)

        return {
            "mean": np.mean(y),
            "std": np.std(y),
            "autocorr": autocorrelation(y),
            "max": np.max(y),
            "min": np.min(y)
        }

    def get_snr_comparisons(self) -> tuple[np.float64, np.float64]:
        def snr(y: ntp.NDArray) -> np.float64:
            signal_power = np.mean(y ** 2)
            noise_power = np.mean((y - np.mean(y)) ** 2)
            return 10 * np.log10(signal_power / noise_power)
        return snr(self.original), snr(self.filtered)

    def get_removed_part_statistics(self) -> dict[str, np.float64]:
        removed = self.original - self.filtered
        return self.__get_signal_statistics(removed)


class NoiseCancelPreprocessor:
    """Cancels noise in data using different methods"""

    def __init__(
            self, k: int,
            data_y_values: ntp.NDArray,
            data_x_values: ntp.NDArray
    ) -> None:
        Validator.assert_shapes_equal(data_x_values, data_y_values)
        assert(k > 0, f"k should be positive, but {k=}")
        assert(
            k <= data_x_values.shape[0],
            f"k should be less than data length, ideally it should be"
            f"not more than 1% of data length, but"
            f"{k=}, data length = {data_x_values.shape[0]}"
        )

        self.k = k
        self.Y = data_y_values
        self.X = data_x_values
        self.__methods = {
            "sliding_mean": self.__sliding_mean,
            "sliding_median": self.__sliding_median,
            "limit": self.__flat_limit,
            "butterworth": self.__butterworth,
            "chebyshev": self.__chebyshev,
            "elliptic": self.__elliptic,
        }
        self.prep_x = self.X
        self.prep_y = self.Y

    def get_result(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        return self.prep_x, self.prep_y

    def preprocess(self, method="sliding_mean", *args) -> None:
        if method in self.__methods:
            self.prep_x, self.prep_y = self.__methods[method](*args)
        else:
            raise ValueError(
                f"Invalid noise cancel method"
                f"Available methods are {self.__methods.keys()}"
            )

    def __rescale_x(self) -> ntp.NDArray:
        return self.prep_x[
               self.k // 2: self.prep_x.shape[0] - self.k // 2 + 1
        ]

    def __sliding_mean(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        def proc(arr: ntp.NDArray, k: int) -> ntp.NDArray:
            n = arr.shape[0]
            running_sum = np.sum(arr[:k])
            averages = np.empty(n - k + 1)
            averages[0] = running_sum / k
            for i in range(1, n - k):
                running_sum = running_sum - arr[i - 1] + arr[i + k - 1]
                averages[i] = running_sum / k
            return averages
        return (
            self.__rescale_x(),
            proc(self.prep_y, self.k)
        )

    def __sliding_median(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        def proc(arr: ntp.NDArray, k: int) -> ntp.NDArray:
            n = arr.shape[0]
            medians = np.empty(n - k + 1)
            medians[0] = np.median(arr[:k])
            for i in range(1, n - k):
                medians[i] = np.median(arr[i:i + k:])
            return medians
        return (
            self.__rescale_x(),
            proc(self.prep_y, self.k)
        )

    def __flat_limit(
            self, limit: np.float64
    ) -> tuple[ntp.NDArray, ntp.NDArray]:
        for i in range(self.prep_y.shape[0]):
            if abs(self.prep_y[i]) < limit:
                self.prep_y[i] = 0
        return self.prep_x, self.prep_y

    def __filter(self, filter_func):
        freq_ratio = (self.k / self.prep_x.shape[0]) * 2
        b, a = filter_func(freq_ratio)
        filtered_y = scipy.signal.lfilter(b, a, self.prep_y)
        return self.X, filtered_y

    def __butterworth(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        return self.__filter(
            lambda freq: scipy.signal.butter(
                4, freq, btype='low', analog=False, fs=None
            )
        )

    def __chebyshev(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        return self.__filter(
            lambda freq: scipy.signal.cheby1(
                4, 1, freq, btype='low', analog=False, fs=None
            )
        )

    def __elliptic(self) -> tuple[ntp.NDArray, ntp.NDArray]:
        return self.__filter(
            lambda freq: scipy.signal.ellip(
                4, 0.5, 30, freq, btype='low', analog=False, fs=None
            )
        )
