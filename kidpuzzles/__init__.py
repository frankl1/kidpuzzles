from gymnasium.envs.registration import register

register(
    id="kidpuzzles/DigitsPuzzleEnv-v0",
    entry_point="kidpuzzles.envs:DigitsPuzzleEnv",
    max_episode_steps=300
)