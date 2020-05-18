[//]: # (Image References)

[image1]: scores_max.png "Max scores"


### Introduction

This report describes the implementation of the Deep Deterministic Poligy Gradient (DDPG) method to solve the Unity Tennis environment. More information on DDPG can be found in the [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

There are a few potential methods suitable for this environment. Proximal Policy Optimization (PPO), Actor-Critic methods (such as Asynchronous Advantage Actor-Critic (AC3)), and DDPG (which can be considered an Actor-Critic Method or an extension of DQN to continuous action spaces) could potentially be used. DDPG requires only a single step to perform an update, whereas PPO requires a trajectory of multiple steps, so DDPG potentially may train faster. DDPG's use of a memory buffer also permits it to utilise decorrelated experiences, ensuring that different states are investigated. AC3 does not use a memory buffer because it assumes that samples from parallel agents will be decorrelated. There are only 2 agents in this environment and they are not collecting experiences in parallel, so AC3 is not suitable.

### Implementation

This task used the DDPG implementation used in the continuos control project and did not require substantial alteration. The neural networks were unaltered and the only hyperparameter changed was the noise multiplier, epsilon and the update frequency. The networks were updated 4 times every step. The hyperparameters used are:
* BUFFER_SIZE = int(1e6)  (replay buffer size)
* BATCH_SIZE = 128        (minibatch size)
* GAMMA = 0.95            (discount factor)
* TAU = 1e-3              (for soft update of target parameters)
* LR_ACTOR = 1e-4         (learning rate of the actor)
* LR_CRITIC = 1e-3        (learning rate of the critic)
* WEIGHT_DECAY = 0        (L2 weight decay)
* UPDATE_EVERY = 1        (how often to update the network)
* EPSILON = 1             (how much to scale the noise)
* EPS_DECAY = 1e-4        (how much to decay epsilon)

The actor and crtic models are defined in model.py and both have 3 fully connected layers. A batch normalisation follows the first layer, which improves training when the state inputs are scaled very differently. The unit sizes are 256 for the first layer and 128 units for the second layer for both networks. In the critic model, the action is concatenated after the first dense layer and batch normalisation. The weights are initialised as recommended by the paper, aside from a difference in scale between the final layers. The actor final layer is initialised with uniform noise between -3e-3 and 3e-3, whereas the critic is initialised with uniform noise between -3e-4 and 3e-4. A smaller neural network, such as those implemented, appears to be easier to train. The size of the networks and the hyperparameters were determined largely through trial and error, adpating from the published hyperparamters, but it was impossible to investigate every possible combination.

Both of the agents in this environment update the same replay buffer and experiences are sampled randomly. Prioritised Experience Replay could potentially be implemented to improve the training process.

Noise is added to the action selection to ensure exploration rather than always following the policy. Thus, this is an off-policy method. However, unlike epsilon-greedy techniques, the stochastic nature of the decision can still be learned. The noise added is taken from the Ornstein-Uhlenbeck process, which describes Brownian motion. This is scaled by epsilon, and decreased throughtout training, so as the agent learns more it explores less.

### Results

The plot below shows the max score of the two agents throughout training. The environment was solved in 488 episodes and remained fairly stable for the following 500 episodes, though the average score did drop slightly below 0.5.

![Scores][image1]


### Potential Future Work
There is a high amount of fluctuation in the scores for this task, presenting an opportunity for improvement. Other approaches could be invesigated for improved stability. For example, [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347.pdf).

This implementation could potentially be improved by using Prioritised Experience Replay or changing the distribution of noise sampled when generating actions.

Distributed algorithms, such as [Distributed Distributional Deep Deterministic Policy Gradient (D4PG)](https://openreview.net/pdf?id=SyZipzbCb) may possibly improve performance, though this implementation trains fairly quickly.


