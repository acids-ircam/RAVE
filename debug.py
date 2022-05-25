#%%
from rave.core import leaf_apply


def fun(x):
    return 2 * x


tree = [1, [2, 3], [[[3, 4, 5], 5]]]

leaf_apply(fun, tree)
# %%
