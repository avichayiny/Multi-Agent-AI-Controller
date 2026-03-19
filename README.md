# AI Multi-Agent Controller: Plant Watering Simulation

## Overview
This repository contains a three-stage AI project developed in Python for a multi-agent resource-management simulation. The core mission remains the same across all stages, but the environment's complexity and uncertainty increase, requiring different Artificial Intelligence approaches.

## The Domain: "Plant Watering"
The simulation takes place on a 2D grid containing autonomous robots, water taps, plants, and walls. 
* **The Goal:** Robots must navigate the grid, load water from taps, and pour it onto plants to meet their specific water requirements.
* **The Challenge:** Robots have limited water-carrying capacities, and as the project progresses, the environment introduces stochastic action failures, randomized rewards, and completely unknown probabilities.

## Project Stages
*(Note: The full assignment instructions and rules are available as PDF files inside each respective folder).*

* **Part 1: Deterministic Environments (Folder: `Part1-Deterministic`)**
  * **Task:** Find the optimal sequence of actions to water all plants.
  * **Algorithms:** Implemented heuristic search algorithms (A*, GBFS) for optimal path-finding and planning.
  
* **Part 2: Stochastic Environments (Folder: `Part2-Stochastic-MDP`)**
  * **Task:** Maximize expected rewards within a strict step horizon. The world is modeled as a Markov Decision Process (MDP) where defective robots have a probability to fail actions, and plants yield random rewards.
  * **Algorithms:** Developed an MDP-based controller to choose the best next action per state.

* **Part 3: Unknown Environments (Folder: `Part3-Reinforcement-Learning`)**
  * **Task:** Optimize strategies dynamically in a world where the robots' success rates and plant reward distributions are completely unknown to the agent.
  * **Algorithms:** Applied Reinforcement Learning (Adaptive Dynamic Programming - ADP) to learn the environment's models on the fly.

## Note on Codebase
The core AI logic, heuristic functions, and RL algorithms were implemented by me (located in the `ex1.py`, `ex2.py`, and `ex3.py` files). The simulation engine, environment infrastructure, and test checkers were provided by the university course staff.
