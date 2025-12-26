# BipedalWalker-v3: PPO Hyperparameter Ablation Study

This repository investigates the impact of various hyperparameters on the training stability and performance of a Proximal Policy Optimization (PPO) agent in the `BipedalWalker-v3` environment.

## Installation

```bash
pip install gymnasium[box2d] stable-baselines3[extra] shimmy
```

![table](table.jpg)

## Usage

- To execute all ablation tests sequentially run train.py

- To load a specific model and its normalization statistics to record a performance video run evaluate.py

- The remaining files starting with eval are all videos of the final performances of the experiments ran

# Here are the results of the greatest training run of 1M timesteps per experiment:
![1Mresults](1M.jpg)

# Here are the results of a previous run of 500K timesteps per experiment
![500Kres](500K.jpg)

# Here is the video of the best model (idk if this is the right video lol it looks slow) that developed a graceful frolic

https://github.com/user-attachments/assets/af6b87bb-0fb4-4c85-b8c2-974a6bdc25ab

# Here are its rewards, showing that it crossed the 300 point mark

![Screenshot 2025-12-25 214211](https://github.com/user-attachments/assets/f8ad6e8b-8e1b-4bca-a557-947be530a495)

# Here are the videos of my other agents
## Low lambda

https://github.com/user-attachments/assets/fe922a11-19c5-42e6-8397-b8b918927afb

## high lambda

https://github.com/user-attachments/assets/7fc55232-0035-40bc-8be7-8f5dbe6717c9

## Normalized baseline

https://github.com/user-attachments/assets/bc1dd2e4-002a-4665-977f-066c3d67702f

## High exploration

https://github.com/user-attachments/assets/aeb60291-8dca-4e55-8805-31e3a160023a

## short sighted

https://github.com/user-attachments/assets/5391c4c8-3dd9-454d-b1f6-5f239003e43a

## strict clipping

https://github.com/user-attachments/assets/5c63d0ec-fe50-4fd2-9e10-4174f917250e



















