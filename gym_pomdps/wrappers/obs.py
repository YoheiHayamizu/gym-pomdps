import gym
from gym import Env
import gym_pomdps
import numpy as np

class DiscreteToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,))
    
    def observation(self, obs):
        new_obs = np.zeros(self.n)
        new_obs[obs] = 1 
        return new_obs


class FlickeringWrapper(gym.Wrapper):
    def __init__(self, env: Env, prob: float = 0.5) -> None:
        gym.Wrapper.__init__(self, env)
        assert (0.0 <= prob <= 1.0), "Should 0.0 < prob < 1.0"  
        self.flickering_prob = prob

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        prob_flicker = np.random.rand()

        if prob_flicker < self.flickering_prob:
            obs = np.zeros_like(obs)

        return obs, reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    env = gym.make("POMDP-rock_sample_5_4-episodic-v2")
    env = DiscreteToBoxWrapper(env)
    T = 100
    s_t = env.reset()
    for t in range(T):
        a_t = env.action_space.sample()
        s_t, r_t, done, info = env.step(a_t)
        print(s_t, len(s_t))
        if done:
            s_t = env.reset()


