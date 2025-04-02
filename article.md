# AI Agent vs 2-Year-Old Kid

## Introduction
In this article, I will present a personal project where I trained an AI agent using reinforcement learning to solve a puzzle my 2-year-old son started solving at 18 months. This project aims to explore the differences in problem-solving approaches between a young child and an AI. 

## The Puzzle: placing the first 10 digits on a board
### Description
The puzzle is composed of the digits from $0$ to $9$ and a board containing 10 slots, one for each of the digits. The slots are shaped with respect to the digit shape such that each digit can only go into its respective slot. Here an image of the puzzle. 
![Wooden Numbers 0-9 Inset Puzzle](https://i.ebayimg.com/images/g/YtkAAOSwiQNlLCOD/s-l1600.webp)

The puzzle starts with all the digits out of the board, and the goal is to place each digit in the right slot on the board. The puzzle is solved when all the digits has been placed correctly on the board as shown by the above image.

### Purpose
The purpose of the game is to help children develop cognitive and motor skills.

### Value for Kids' Development
Playing this game helps children:
- Improve hand-eye coordination
- Develop problem-solving skills
- Enhance spatial awareness
- Recognize shapes
- Learn counting 

## Building the AI Agent
### Setting Up the Environment
    
  I used [Gymnasium](https://gymnasium.farama.org/) to create the game environment. The environment is named **DigitsPuzzleEnv** and is the first environment in the [KidPuzzle](https://github.com/frankl1/kidpuzzles) package which is ready to welcome other kid puzzle environments. The package is public on Github and contributions to enrich it are very welcome. 
  

### Training the AI Agent
    
  I used [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) to train an AI agent on DigitsPuzzleEnv. The agent's goal is to learn how to solve the puzzle as fast as possible. For this purpose, I designed a custom Manhattan distance-based reward function which encourages action such as moving digits inside the targets area and moving a digit closer to its target location while penalizing other actions such as moving a digit out of the board or getting a digit off the target area.

## Demonstrations
- **Kid Playing the Game**
  - [Insert video of the 2-year-old kid playing the game]
- **AI Agent Playing the Game**
  
  The following video shows the agent trying to solve the puzzle after $600$ episodes of training.
  <video controls src="videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-600.mp4" title="Agent solving the puzzle after 600 episodes of training"></video>

  It can be seen that the agent still struggles a lot, its behavior is quite random at this point of its learning process. We have to wait for about $4000$ training episodes to see an improvement of the agent capabilities.
  <video controls src="videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-4000.mp4" title="Agent solving the puzzle after 4000 episodes of training"></video>.

  Despite the agent is still not able to solve the puzzle completely, it manages to place the digits $2, 3, 4, 5$ and $6$ at their target locations. The remaining digits are 1 step away from their targets except for digit $9$ which is 2 steps away. This is already pretty good.

  A descent agent, which have mastered the puzzle is obtained after $10200$ episodes of training. At this point, the agent is able to solve the puzzle with the minimum number of steps in just 10 seconds.
  <video controls src="videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-10200.mp4" title="Agent solving the puzzle after 10200 episodes of training"></video>


## Conclusion
This project highlights the differences in problem-solving approaches between a young child and an AI agent. While the child relies on intuition and trial-and-error, the AI agent uses reinforcement learning to optimize its performance over time.

Thank you for reading! Feel free to share your thoughts and feedback.
