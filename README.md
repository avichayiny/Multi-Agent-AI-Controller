# 🤖 AI Multi-Agent: Resource Management

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Artificial Intelligence](https://img.shields.io/badge/AI-FF9900?style=for-the-badge&logo=openai&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-000000?style=for-the-badge)

A three-stage Artificial Intelligence project demonstrating the evolution of decision-making algorithms in autonomous agents. The system focuses on solving a logical resource-management problem (plant watering) under increasingly complex and unpredictable mathematical models.

## 🌍 The Domain & Challenge
Autonomous robots inhabit a logical 2D grid containing water taps, plants, and obstacles. The primary objective is to output an optimal policy—a mathematical sequence of actions or state-action mappings—to successfully manage resources.

The project is structured to demonstrate how different AI paradigms handle computational uncertainty:
* **Stage 1:** Perfect knowledge and predictable outcomes.
* **Stage 2:** Known probabilities with stochastic state transitions.
* **Stage 3:** Completely unknown environment requiring learning on the fly.

## 🧠 AI Evolution Stages

### Part 1: Deterministic Environments
* **Task:** Calculate and output the optimal, guaranteed sequence of actions to complete the mission.
* **Algorithms Implemented:** Heuristic search algorithms including **A* (A-Star)** and **Greedy Best-First Search (GBFS)** for state-space path-finding.
* **Location:** `/Part1-Deterministic`

### Part 2: Stochastic Environments (MDP)
* **Task:** Maximize expected rewards within a strict step limit. Actions have a probability of failure, and rewards are randomized.
* **Algorithms Implemented:** Modeled the world as a **Markov Decision Process (MDP)**, calculating mathematical policies to choose the best next action per state.
* **Location:** `/Part2-Stochastic-MDP`

### Part 3: Unknown Environments (Reinforcement Learning)
* **Task:** Dynamically optimize strategies in a world where success rates and reward distributions are completely hidden from the agent.
* **Algorithms Implemented:** **Reinforcement Learning**, specifically **Adaptive Dynamic Programming (ADP)**, allowing the agent to learn the environment's model through continuous state exploration and exploitation.
* **Location:** `/Part3-Reinforcement-Learning`

## 🛠️ Implementation Note
The core AI algorithms, heuristic functions, state evaluations, and RL logic (`ex1.py`, `ex2.py`, `ex3.py`) were developed entirely from scratch by me. The environment infrastructure, mathematical models, and automated test checkers were provided by the university.
