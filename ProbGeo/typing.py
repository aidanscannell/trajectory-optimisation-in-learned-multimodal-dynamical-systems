from jax import numpy as np
from typing import Tuple

from ProbGeo.kernels import DiffRBF

MeanAndVariance = Tuple[np.array, np.array]
InputData = np.array
OutputData = np.array
MeanFunc = np.float64
Kernel = DiffRBF
# TODO replace DiffRBF with abstract kernel class
