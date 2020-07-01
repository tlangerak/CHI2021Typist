from gym.envs.registration import register

register(
    id='Finger-v1',
    entry_point='custom_envs.FingerEnv:FingerEnv',
    max_episode_steps=200,
)