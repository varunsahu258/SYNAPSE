from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


@dataclass(frozen=True)
class DirectorObservation:
    inequality: float
    crime: float
    regional_stability: float
    resource_distribution: float
    unemployment: float
    social_satisfaction: float
    neighborhood_disparity: float

    def as_array(self) -> list[float]:
        return [self.inequality, self.crime, self.regional_stability, self.resource_distribution, self.unemployment, self.social_satisfaction, self.neighborhood_disparity]


@dataclass(frozen=True)
class DirectorAction:
    taxation: float
    subsidies: float
    policing: float
    welfare_allocation: float
    zoning_adjustments: float
    resource_distribution: float

    def as_array(self) -> list[float]:
        return [self.taxation, self.subsidies, self.policing, self.welfare_allocation, self.zoning_adjustments, self.resource_distribution]

    @classmethod
    def from_array(cls, values: list[float]) -> "DirectorAction":
        clipped = [_clamp(v, -1.0, 1.0) for v in values]
        return cls(*[float(v) for v in clipped])


@dataclass(frozen=True)
class RewardWeights:
    stability: float = 1.8
    inequality_penalty: float = 1.3
    suffering_penalty: float = 1.4
    crime_penalty: float = 1.2
    sustainability: float = 1.0


class TransitionModel(Protocol):
    def step(self, observation: DirectorObservation, action: DirectorAction) -> DirectorObservation: ...


@dataclass
class LinearGovernanceTransition:
    noise_scale: float = 0.005
    rng: random.Random = field(default_factory=random.Random)

    def step(self, observation: DirectorObservation, action: DirectorAction) -> DirectorObservation:
        n = lambda: self.rng.gauss(0.0, self.noise_scale)
        inequality = _clamp01(observation.inequality - 0.07 * action.taxation - 0.05 * action.welfare_allocation + 0.04 * action.zoning_adjustments + n())
        crime = _clamp01(observation.crime - 0.08 * action.policing - 0.03 * action.welfare_allocation + 0.04 * observation.unemployment + n())
        unemployment = _clamp01(observation.unemployment - 0.05 * action.subsidies + 0.03 * action.taxation + n())
        resource_distribution = _clamp01(observation.resource_distribution + 0.05 * action.resource_distribution - 0.02 * observation.neighborhood_disparity + n())
        social_satisfaction = _clamp01(observation.social_satisfaction + 0.04 * action.welfare_allocation + 0.03 * action.subsidies - 0.04 * crime + n())
        neighborhood_disparity = _clamp01(observation.neighborhood_disparity - 0.05 * action.zoning_adjustments - 0.03 * action.resource_distribution + 0.02 * inequality + n())
        regional_stability = _clamp01(observation.regional_stability + 0.05 * social_satisfaction - 0.05 * crime - 0.04 * inequality + n())
        return DirectorObservation(inequality, crime, regional_stability, resource_distribution, unemployment, social_satisfaction, neighborhood_disparity)


def compute_reward(observation: DirectorObservation, weights: RewardWeights) -> float:
    suffering = (1.0 - observation.social_satisfaction) + observation.unemployment
    sustainability = 0.6 * observation.resource_distribution + 0.4 * (1.0 - observation.neighborhood_disparity)
    return float(weights.stability * observation.regional_stability - weights.inequality_penalty * observation.inequality - weights.suffering_penalty * suffering - weights.crime_penalty * observation.crime + weights.sustainability * sustainability)


class DirectorGymEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, *, transition_model: TransitionModel | None = None, reward_weights: RewardWeights | None = None, episode_length: int = 100, seed: int | None = None) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for DirectorGymEnv")
        super().__init__()
        self.transition_model = transition_model or LinearGovernanceTransition()
        self.reward_weights = reward_weights or RewardWeights()
        self.episode_length = episode_length
        self._rng = random.Random(seed)
        self._step_count = 0
        self._state = self._sample_initial_observation()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=float)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = random.Random(seed)
        self._step_count = 0
        self._state = self._sample_initial_observation(options)
        return self._state.as_array(), {}

    def step(self, action: list[float]):
        self._step_count += 1
        self._state = self.transition_model.step(self._state, DirectorAction.from_array(list(action)))
        reward = compute_reward(self._state, self.reward_weights)
        terminated = self._step_count >= self.episode_length
        return self._state.as_array(), reward, terminated, False, {"observation": self._state, "metrics": evaluate_observation(self._state)}

    def _sample_initial_observation(self, options: Mapping[str, Any] | None = None) -> DirectorObservation:
        base = {
            "inequality": self._rng.uniform(0.2, 0.7),
            "crime": self._rng.uniform(0.1, 0.6),
            "regional_stability": self._rng.uniform(0.2, 0.8),
            "resource_distribution": self._rng.uniform(0.2, 0.8),
            "unemployment": self._rng.uniform(0.1, 0.5),
            "social_satisfaction": self._rng.uniform(0.2, 0.8),
            "neighborhood_disparity": self._rng.uniform(0.1, 0.7),
        }
        if options:
            for k, v in options.items():
                if k in base:
                    base[k] = _clamp01(float(v))
        return DirectorObservation(**base)


def evaluate_observation(observation: DirectorObservation) -> dict[str, float]:
    return {
        "inequality": observation.inequality,
        "crime": observation.crime,
        "stability": observation.regional_stability,
        "survival_rate": _clamp01(0.5 * observation.regional_stability + 0.5 * observation.social_satisfaction),
        "resource_efficiency": _clamp01(0.7 * observation.resource_distribution + 0.3 * (1.0 - observation.neighborhood_disparity)),
    }


def run_baseline_rollout(env: DirectorGymEnv, *, policy: str, steps: int) -> dict[str, float]:
    env.reset()
    records=[]
    for _ in range(steps):
        if policy=="do_nothing":
            action=[0.0]*6
        elif policy=="safety_first":
            action=[0.3,0.2,0.9,0.4,0.1,0.3]
        else:
            raise ValueError(policy)
        _,_,done,_,info=env.step(action)
        records.append(info["metrics"])
        if done: break
    return _aggregate(records)


def train_director_ppo(output_dir: str | Path, *, total_timesteps: int = 20000, episode_length: int = 100, seed: int = 42, reward_weights: RewardWeights | None = None) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    run_dir=Path(output_dir); run_dir.mkdir(parents=True, exist_ok=True)
    env=DirectorGymEnv(episode_length=episode_length, seed=seed, reward_weights=reward_weights)
    model=PPO("MlpPolicy", env, verbose=0, tensorboard_log=str(run_dir/"tensorboard"), seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=CheckpointCallback(save_freq=max(1000, episode_length), save_path=str(run_dir/"checkpoints"), name_prefix="director_ppo"))
    path=run_dir/"director_ppo_final.zip"; model.save(str(path)); return path


def evaluate_trained_model(model_path: str | Path, *, episodes: int = 3, episode_length: int = 100) -> dict[str, float]:
    from stable_baselines3 import PPO
    env=DirectorGymEnv(episode_length=episode_length)
    model=PPO.load(str(model_path))
    episode_metrics=[]
    for _ in range(episodes):
        obs,_=env.reset(); done=False; metrics=[]
        while not done:
            action,_=model.predict(obs, deterministic=True)
            obs,_,term,trunc,info=env.step(list(action)); done=term or trunc; metrics.append(info["metrics"])
        episode_metrics.append(_aggregate(metrics))
    return _aggregate(episode_metrics)


def _aggregate(rows: list[dict[str,float]]) -> dict[str,float]:
    if not rows:
        return {"inequality":0.0,"crime":0.0,"stability":0.0,"survival_rate":0.0,"resource_efficiency":0.0}
    keys=rows[0].keys()
    return {k: sum(r[k] for r in rows)/len(rows) for k in keys}

def _clamp(v: float, lo: float, hi: float) -> float: return max(lo, min(hi, float(v)))
def _clamp01(v: float) -> float: return _clamp(v,0.0,1.0)
