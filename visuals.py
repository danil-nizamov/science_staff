import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as ntp
from validator import Validator


class Visualizer:

    def __init__(
            self,
            data_x_values: ntp.NDArray,
            data_y_values: ntp.NDArray,
            x_label: str,
            y_label: str
    ) -> None:
        Validator.assert_shapes_equal(data_x_values, data_y_values)
        self.X = data_x_values
        self.Y = data_y_values
        self.x_label = x_label
        self.y_label = y_label
        self.__methods = {
            "batches": self.__by_batches,
        }

    def draw(self, method="batches", *args):
        if method in self.__methods:
            return self.__methods[method](*args)
        else:
            raise ValueError(
                f"Invalid visualization method"
                f"Available methods are {self.__methods.keys()}"
            )

    def __get_min_and_max_y(self) -> tuple[int, int]:
        mn = self.Y[0]
        mx = self.Y[0]
        for item in self.Y:
            if item > mx:
                mx = item
            if item < mn:
                mn = item
        return mn, mx

    def __by_batches(self, n_batches: int) -> None:
        batch_size = self.X.shape[0] // n_batches
        y_min, y_max = self.__get_min_and_max_y()
        for i in range(n_batches):
            plt.figure(figsize=(12, 3))
            start_index = i * batch_size
            if i < n_batches - 1:
                end_index = (i + 1) * batch_size
            else:
                end_index = len(self.X)
            plt.plot(
                self.X[start_index:end_index],
                self.Y[start_index:end_index]
            )
            plt.title(f'Batch {i + 1}')
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            plt.ylim(y_min, y_max)
            plt.show(block=False)
        plt.show()
