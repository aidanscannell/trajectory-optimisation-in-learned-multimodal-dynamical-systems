class FakeGP:
    from ProbGeo.utils.gp import load_data_and_init_kernel_fake

    X, Y, kernel = load_data_and_init_kernel_fake(
        filename="./saved_gp_models/params_fake.npz"
    )
    mean_func = 0.0
    q_sqrt = None
