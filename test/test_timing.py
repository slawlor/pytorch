# import torch
from torch.utils.benchmark import Timer
# import pickle

timer = Timer(
    stmt="x + y",
    setup="""
        x = torch.ones((16,))
        y = torch.ones((16,))
    """
)

stats: CallgrindStats = timer.collect_callgrind()
print(stats)


# add_after = Timer(
#     "c = torch.add(a, b);",
#     """
#         a = torch.ones(1);
#         b = torch.ones(1);
#     """,
# ).collect_callgrind().as_standardized().stats(inclusive=False)

# with open('add_after.pickle', 'wb') as after_f:
#     pickle.dump(add_after, after_f)
