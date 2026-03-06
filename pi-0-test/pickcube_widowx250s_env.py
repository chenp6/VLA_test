from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv, PICK_CUBE_DOC_STRING
from mani_skill.utils.registration import register_env



@register_env("PickCubeWidowX250S-v1", max_episode_steps=50)
class PickCubeWidowX250SEnv(PickCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowx250s", **kwargs)


PickCubeWidowX250SEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="WidowX250S")

