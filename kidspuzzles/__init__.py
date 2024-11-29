from gymnasium.envs.registration import register

register(
    id="kidspuzzles/DigitsPuzzleEnv-v0",
    entry_point="kidspuzzles.envs:DigitsPuzzleEnv",
    max_episode_steps=300
)