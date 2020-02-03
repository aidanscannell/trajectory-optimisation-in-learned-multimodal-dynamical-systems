import jax
from jax import batching, jacfwd, jvp, lu, partial, tree_map, vmap
from jax.api import (_argnums_partial, _check_real_input_jacfwd, _std_basis,
                     _unravel_array_into_pytree)


def _jacfwd(fun, argnums=0, holomorphic=False, return_value=False):
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = _argnums_partial(f, argnums, args)
        holomorphic or tree_map(_check_real_input_jacfwd, dyn_args)
        pushfwd = partial(jvp, f_partial, dyn_args)
        y, jac = vmap(pushfwd,
                      out_axes=(None, batching.last))(_std_basis(dyn_args))
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac = tree_map(partial(_unravel_array_into_pytree, example_args, -1),
                       jac)
        if return_value:
            return jac, y
        else:
            return jac

    return jacfun


def jacfwd(fun, argnums=0, holomorphic=False):
    """Jacobian of `fun` evaluated column-by-column using forward-mode AD.
@@ -462,17 +479,47 @@ def jacfwd(fun, argnums=0, holomorphic=False):
   [ 0.        , 16.        , -2.        ],
   [ 1.6209068 ,  0.        ,  0.84147096]]
  """
    return _jacfwd(fun, argnums, holomorphic, return_value=False)


def value_and_jacfwd(fun, argnums=0, holomorphic=False):
    """Creates a function which evaluates both `fun` and the Jacobian of `fun`.
  The Jacobian of `fun` is evaluated column-by-column using forward-mode AD.
  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default `0`).
    holomorphic: Optional, bool. Indicates whether `fun` is promised to be
      holomorphic. Default False.
  Returns:
    A function with the same arguments as `fun`, that evaluates both `fun` and
    the Jacobian of `fun` using forward-mode automatic differentiation, and
    returns them as a two-element tuple `(val, jac)` where `val` is the
    value of `fun` and `jac` is the Jacobian of `fun`.
  >>> def f(x):
  ...   return jax.numpy.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
  ...
  ... val, jac = jax.value_and_jacfwd(f)(np.array([1., 2., 3.])))
  ...
  >>> print(val)
  [ 1.         15.         10.          2.52441295]
  >>> print(jac)
  [[ 1.        ,  0.        ,  0.        ],
   [ 0.        ,  0.        ,  5.        ],
   [ 0.        , 16.        , -2.        ],
   [ 1.6209068 ,  0.        ,  0.84147096]]
  """
    jacfun = _jacfwd(fun, argnums, holomorphic, return_value=True)

    def value_and_jacobian_func(*args, **kwargs):
        jacobian, value = jacfun(*args, **kwargs)
        return value, jacobian

    return value_and_jacobian_func
