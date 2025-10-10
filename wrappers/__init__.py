# from samplers.goalonly import GoalOnly
# from samplers.norndgcfql import NoRNDGCFQLSampler
# from samplers.rndgcfql import RNDGCFQLSampler

# samplers = dict(
#     norndgcfql=NoRNDGCFQLSampler,
#     goalonly=GoalOnly,
#     rndgcfql=RNDGCFQLSampler,
# )

# from .combine_with import CombineWith
# from .randomsteps import RandomSteps
# from .ogbench import OGBench
# from .withsubgoal import WithSubgoal
# from .withrnd import WithRND
# from .rndandsubgoal import RNDAndSubgoal

# datafuncs = dict(
#     combine_with=CombineWith,
#     randomsteps=RandomSteps,
#     ogbench=OGBench,
#     withsubgoal=WithSubgoal,
#     withrnd=WithRND,
#     rndandsubgoal=RNDAndSubgoal,
# )

from .withrnd import WithRND
from .randomsteps import RandomSteps
from .rndsubgoals import RNDSubgoals

wrappers = dict(
    withrnd=WithRND,
    randomsteps=RandomSteps,
    rndsubgoals=RNDSubgoals,
)