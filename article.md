# From Playtime to Programming: An AI Adventure Inspired by My Son

## Introduction
This project began not in a lab, but on the living room floor, watching my son engages with a simple digit puzzle. His playtime sparked a question: could I teach an AI to master this same challenge? In this article, I present my journey in training an AI agent using reinforcement learning to solve the digit puzzle. My son, now two years old, has been captivated by this digit puzzle since he was just 18 months old. Watching his progress inspired me to embark on this project. 

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

Translating the intuitive actions of a child at play into a set of rules and rewards for an AI agent was a fascinating challenge. For instance, I could not reproduce my son's reward model as it is too complex and seems to be changing overtime.

### Setting Up the Environment
    
I used [Gymnasium](https://gymnasium.farama.org/) to create the game environment. The environment is named **DigitsPuzzleEnv** and is the first environment in the [KidPuzzle](https://github.com/frankl1/kidpuzzles) package which is ready to welcome other kid puzzle environments. The package is public on Github and contributions to enrich it are very welcome. 
  

### Training the AI Agent
    
I used [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) to train an AI agent on DigitsPuzzleEnv. The agent's goal is to learn how to solve the puzzle as fast as possible. For this purpose, I designed a custom Manhattan distance-based reward function which encourages action such as moving digits inside the targets area and moving a digit closer to its target location while penalizing other actions such as moving a digit out of the board or getting a digit off the target area.

## Demonstrations
### Kid Playing the Game

My two-year-old kid takes about 1 minutes and 15 seconds to solve the puzzle. The following video shows him completing a game.
<video controls src="./videos/2yo-solving-digits.mp4" title="Two-year-old solving the puzzle"></video>
It used to be difficult for him to differentiate between 6 and 9. Now, his main challenge is to know which side of 8 should be up and which one should be down.

### AI Agent Playing the Game
  
The following video shows the agent trying to solve the puzzle after $600$ episodes of training.
<video controls src="./videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-600.mp4" title="Agent solving the puzzle after 600 episodes of training"></video>

At this early stage, the agent's movements are erratic, like a toddler taking its first steps. It is clear that it is still exploring the environment, trying to grasp the rules of the game. We have to wait for about $4000$ training episodes to see an improvement of the agent capabilities.
<video controls src="./videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-4000.mp4" title="Agent solving the puzzle after 4000 episodes of training"></video>.

Despite the agent is still not able to solve the puzzle completely. It's no longer just flailing randomly; it's starting to make deliberate choices, even if they're not always the right ones. It manages to place the digits $2, 3, 4, 5$ and $6$ at their target locations. The remaining digits are 1 step away from their targets except for digit $9$ which is 2 steps away. This is already pretty good.

A descent agent, which have mastered the puzzle is obtained after $10200$ episodes of training. Its movements are precise and efficient, a far cry from its initial clumsy attempts. At this point, the agent is able to solve the puzzle with the minimum number of steps in just 10 seconds.
<video controls src="./videos/DigitsPuzzle-10-step_f0.5_i0.33/training-episode-10200.mp4" title="Agent solving the puzzle after 10200 episodes of training"></video>


## Conclusion: More Than Just a Puzzle - A Glimpse into Learning
This project started as a fun way to explore AI and reinforcement learning, inspired by my son's own journey with this digit puzzle. Watching him develop his skills, from initially struggling to confidently placing each digit, was a joy. Then, seeing the AI agent go through its own learning process, from clumsy attempts to a surprisingly efficient solution, was equally fascinating.

While the agent ultimately surpassed my son in speed, it's important to remember that the agent's learning is fundamentally different. It's a process of optimization, driven by a reward function. My son's learning is richer, more exploratory, and deeply connected to his overall development.

This project has been a rewarding experience, offering a unique lens through which to view both human and artificial intelligence. It has also sparked my curiosity about the potential of AI to assist in education and to help us better understand the intricacies of human learning. I hope this article has been as engaging for you to read as it was for me to create.

Thank you for reading! Feel free to share your thoughts and feedback.
