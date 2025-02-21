# KidPuzzles 🧩

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/frankl1/kidpuzzles?style=social)
![GitHub forks](https://img.shields.io/github/forks/frankl1/kidpuzzles?style=social)

Welcome to **KidPuzzles**! This repository contains a set of reinforcement learning environments designed to train agents on puzzles tailored for kids. Our goal is to make learning fun and engaging while leveraging the power of reinforcement learning.

## 🎯 Features

- **🎮 DigitsPuzzleEnv** - Interactive environment for number placement puzzles (0-9)
- **🧠 RL-Ready** - Compatible with major reinforcement learning frameworks
- **📊 Visualization** - Built-in rendering for puzzle states
- **📦 Easy Integration** - Simple Gym-style API for seamless implementation
- **🛠️ Easy to Use**: Kick-off with [playground.ipnb](./playground.ipynb)
- **🔄 Extensible**: Easily add new puzzles and environments to expand the learning opportunities.

## 📦 Installation

To get started with KidPuzzles, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/frankl1/kidpuzzles.git
   cd kidpuzzles

2. **Install Dependencies**:

    ```bash 
    pip install -r requirements.txt
    ```

## 📚 Usage
    ```python
    import gymnasium as gym
    import kidpuzzles

    env = gym.make('kidpuzzles/DigitsPuzzleEnv-v0', render_mode = 'human')

    observations = env.reset()

    for _ in range(3):
        action = env.action_space.sample()
        observations, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            observations = env.reset()

    env.close()
    ```

Check out the [playground.ipnb](./playground.ipynb) notebook to see a complete RL training and inference loop on the DigitPuzzleEnv. This notebook provides a step-by-step guide to interacting with the environment and training agents.

## 🏗️ Environment Details
### DigitsPuzzleEnv
- **Goal**: Place the digits 0 to 9 in the correct positions on the board.
- **Actions**: left, right, up, down for each digit.
- **Observations**: The current state of the board and the positions of the digits.
- **Rewards**: Manhattan distance-based reward coupled with bonuses/maluses for reaching intermediate targets.

## 🛠️ Contributing
We welcome contributions from the community! If you have an idea for a new puzzle or want to improve an existing one, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
If you have any questions or suggestions, feel free to open an issue or contact us directly.

🌟 **Star this repository** if you find it useful! 🌟

Happy Puzzling! 🧩🧒