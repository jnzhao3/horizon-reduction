# from samplers.goalonly import GoalOnly
# from samplers.norndgcfql import NoRNDGCFQLSampler
# from samplers.rndgcfql import RNDGCFQLSampler

# samplers = dict(
#     norndgcfql=NoRNDGCFQLSampler,
#     goalonly=GoalOnly,
#     rndgcfql=RNDGCFQLSampler,
# )

from .combine_with import CombineWith

datafuncs = dict(
    combine_with=CombineWith,
)