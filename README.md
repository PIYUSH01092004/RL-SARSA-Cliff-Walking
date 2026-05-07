> **On-Policy Reinforcement Learning: SARSA Implementation on CliffWalking-v1**
> A clean, end-to-end implementation of the SARSA (State-Action-Reward-State-Action) algorithm using Python and Gymnasium. This project demonstrates on-policy temporal-difference learning, $\epsilon$-greedy exploration, and Q-table optimization to solve the classic gridworld "Cliff Walking" problem.

 **Detailed Project Overview**
This project implements the **SARSA** algorithm to navigate the **CliffWalking-v1** environment. Unlike off-policy methods like Q-Learning, SARSA updates its action-value function based on the actual actions taken by the current policy, including exploration steps.

**Key Technical Details**
* **Algorithm**: SARSA (On-Policy Temporal Difference Learning).
* **Environment**: `CliffWalking-v1` from the Farama Foundation's **Gymnasium** library.
* **State Space**: Discrete(48) representing a $4 \times 12$ grid.
* **Action Space**: Discrete(4) (Up, Right, Down, Left).
* **Hyperparameters**:
    * **Learning Rate ($\alpha$):** 0.5.
    * **Discount Factor ($\gamma$):** 0.99.
    * **Exploration Rate ($\epsilon$):** 0.1 ($\epsilon$-greedy policy).
    * **Total Episodes:** 500.

 **Implementation Highlights**
* **Q-Table Initialization**: Utilizes a $48 \times 4$ NumPy array to store and update state-action values.
* **Epsilon-Greedy Strategy**: Balances exploration (sampling random actions) and exploitation (choosing the best-known action).
* **SARSA Update Rule**: Implements the Bellman equation update:  
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$$.
* **Visualization**: Includes a performance evaluation script that renders the trained agent's final path in a human-readable GUI using `pygame-ce`.

**Results**
* **Learning Curve**: The agent successfully learns to avoid the -100 penalty cliff, optimizing from a highly negative initial reward to a stable total reward of approximately **-17** over 17 steps.
* **Convergence**: Demonstrates clear convergence in both total reward and episode length as training progresses.

1. Install dependencies: `pip install gymnasium[toy-text] numpy pygame-ce`.
2. Open `SARSA.ipynb` and run all cells to train and visualize the agent.
