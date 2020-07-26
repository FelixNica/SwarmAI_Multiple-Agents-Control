# Multiple entity logistics and continuous space coordination using deep reinforcement learning

Quick preview:
![](evaluation_loop.gif)

### Problem definition:
Given a number of autonomous mobility entities create a single neural network that can infer their next required positions in a specific logistic reward based environment.<br />

### Implementation details:
The environment consists of a number of <strong>"Bots"</strong>, <strong>"Packs"</strong> and <strong>"Places"</strong> located at different positions in a <strong>continuous</strong> space.<br /><br />
![](bot.png)<br />
"Bot" - representation of an autonomous vehicle, can be a car(Uber/Robotaxi), drone, construction robot or any entity that can move to a given position in order to accomplish a task.<br /><br />
![](pack.png)<br />
"Pack" - representation of cargo, a person, construction material or anything that needs to be transported to a certain destination.<br /><br />
![](place.png)<br />
"Place" - where the pack needs to end up, the destination.<br /><br />
Bots can pick up packs when they are in close proximity to them.<br />
Bots can drop packs when they are in close proximity to the corresponding places.<br />
"Heading" - coordinate vector that provides the bots their new destination.<br />
"Swarm agent" - A single policy neural network that generates a new heading for any specific state of the environment in order to complete the task.<br />
During training the swarm agent receives 50 points reward if a bot picks up a pack and 100 points reward if the pack is delivered to the corresponding place, ex: for 2 bots, 2 packs and 1 place the maximum total reward is 300.<br />
