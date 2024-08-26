from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg

@configclass
class SelfPlayConfig(DirectRLEnvConfig):
    # Simulation
    sim: SimulationCfg = None
    # Robot
    robot_cfg: ArticulationCfg = None
    # Scene
    scene: InteractiveSceneCfg = None
    # Env
    decimation = 
    # Task-specific