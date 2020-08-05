# Multiple entity logistics and continuous space coordination using deep reinforcement learning

Quick preview:<br />
![](evaluation_loop.gif)

### Problem definition:
Given a number of autonomous mobility entities create a single neural network that can infer their next required positions in order to solve a reward based environment.<br /><br />
The environment consists of a number of <strong>"Bots"</strong>, <strong>"Packs"</strong> and <strong>"Places"</strong> located at different positions in a <strong>continuous</strong> space.<br /><br />
![](legend.jpg)<br/><br/>

Bots can pick up packs when they are in close proximity to them.<br />
Bots can drop packs when they are in close proximity to the corresponding places.<br />
"Heading" - coordinate vector that provides the bots their new destination. The bots can navigate to the heading coordinates autonomously.<br />
"Swarm agent" - A single policy neural network that generates a new heading for any specific state of the environment in order to complete the task.<br />
During training the swarm agent receives 50 points reward if a bot picks up a pack and 100 points reward if the pack is delivered to the corresponding place, ex: for 2 bots, 2 packs and 1 place the maximum total reward is 300.<br /><br />
The problem is considered solved if the swarm agent can obtain a reward average over 95% of maximum total reward over a span of 100 episodes

### Policy network implementation:

The input to the policy network consists of a stack of consecutive environment states.<br />
The environment state consists of positional and logistic values:
- x,y position values for bots
- "bot full" flags, 0 or 1 if full
- x,y position values for packs
- "loaded in" pack logistic index (bot index)
- "unloaded in" pack logistic index (place index)
- x,y position values for places
- heading values form the previous state (logistic "memory")

The policy output is the next heading vector consisting of x,y coordinates for all bots.

### Training the policy network

The policy network is trained using "Twin Delayed Deep Deterministic Policy Gradients" - TD3, an actor-critic method 
derived from "Deep Deterministic Policy Gradients" - DDPG that uses a second critic network in order to prevent
value over estimation.<br/>
Original TD3 Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)
