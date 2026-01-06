# from samplers.goalonly import GoalOnly
# from samplers.norndgcfql import NoRNDGCFQLSampler
# from samplers.rndgcfql import RNDGCFQLSampler

# samplers = dict(
#     norndgcfql=NoRNDGCFQLSampler,
#     goalonly=GoalOnly,
#     rndgcfql=RNDGCFQLSampler,
# )

from .combine_with import CombineWith
from .randomsteps import RandomSteps
from .ogbench import OGBench
from .withsubgoal import WithSubgoal
from .withrnd import WithRND
from .rndandsubgoal import RNDAndSubgoal
from .randomsubgoals import RandomSubgoals

datafuncs = dict(
    combine_with=CombineWith,
    randomsteps=RandomSteps,
    ogbench=OGBench,
    withsubgoal=WithSubgoal,
    withrnd=WithRND,
    rndandsubgoal=RNDAndSubgoal,
    randomsubgoals=RandomSubgoals
)