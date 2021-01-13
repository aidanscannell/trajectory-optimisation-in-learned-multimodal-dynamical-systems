# import ProbGeo.gp.gp as gp

from .gp import gp_jacobian, gp_predict
from .utils import (load_data_and_init_kernel_fake,
                    load_data_and_init_kernel_mogpe,
                    load_data_and_init_kernel_sparse)

# from ProbGeo.gp import gp, gp_predict
