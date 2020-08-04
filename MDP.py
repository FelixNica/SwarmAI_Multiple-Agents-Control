import numpy as np
import torch
import time
import TD3
import SwarmBots


def train_policy(hyper_parameters_dict, path):
	param = hyper_parameters_dict

	env = SwarmBots.Game(
		int(param['bots_number']),
		int(param['packs_number']),
		int(param['places_number']),
		float(param['bot_scale']),
		float(param['pack_scale']),
		float(param['place_scale']),
		float(param['load_reward']),
		float(param['unload_reward']),
		int(param['episode_steps']))

	eval_length = int(param['eval_length'])
	eval_freq = int(param['eval_freq'])
	max_steps = int(param['max_steps'])
	start_step = int(param['start_step'])
	batch_size = int(param['batch_size'])
	expl_noise = float(param['expl_noise'])
	min_score = float(param['min_performance']) * env.max_reward

	seed = int(param['seed'])
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	stack_number = 3 #frame stacker implementation - move it to param dict
	frame_dim = env.state_shape
	state_dim = frame_dim * stack_number
	action_dim = env.action_shape
	max_action = env.space_size
	kwargs = {
		'state_dim': int(state_dim),
		'action_dim': int(action_dim),
		'max_action': int(max_action),
		'actor_width': int(param['policy_width']),
		'critic_width': int(param['policy_width']),
		'learning_rate': float(param['learning_rate']),
		'update_rate': int(param['update_rate']),
		'update_tau': float(param['update_tau']),
		'discount': float(param['discount']),
		'policy_noise': float(param['policy_noise']) * max_action,
		'noise_clip': float(param['noise_clip']) * max_action}
	policy = TD3.TD3(**kwargs)
	#policy.load(f"{path}/model")

	frame_stacker = FrameStacker(frame_dim, stack_number)
	replay_buffer = ReplayBuffer(state_dim, action_dim)

	evaluations = [eval_policy(policy, env, seed, eval_length, (frame_dim, stack_number))]

	frame = env.reset()
	state = frame_stacker.make_next_state(frame)

	done = False
	episode_reward = 0
	episode_steps = 0
	episode_num = 0

	for t in range(int(max_steps)):
		episode_steps += 1

		# Select action randomly or according to policy
		if t < start_step:
			action = env.generate_action()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * expl_noise, size=action_dim)
			).clip(0, max_action)

		# Perform action
		next_frame, reward, done = env.step(action)
		done_bool = float(done) if episode_steps < env.episode_steps else 0

		# Build next_state form stacker
		next_state = frame_stacker.make_next_state(next_frame)
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= start_step:
			policy.train(replay_buffer, batch_size)

		if done: 
			# +1 to account for 0 indexing
			print(f"Total T: {t+1} Episode Num: {episode_num+1} "
									f"Episode T: {episode_steps} "
									f"Reward: {episode_reward:.3f}")
			print(80*'_')
			# Reset environment
			frame = env.reset()
			state = frame_stacker.make_next_state(frame)
			frame_stacker.reset()
			done = False
			episode_reward = 0
			episode_steps = 0
			episode_num += 1

		# Evaluate phase
		if (t + 1) % eval_freq == 0:
			evaluations.append(eval_policy(policy, env, seed, eval_length, (frame_dim, stack_number)))
			np.save(f"{path}/metrics", evaluations)
			policy.save(f"{path}/model")
			if evaluations[-1] > min_score:
				big_eval = eval_policy(policy, env, seed, eval_length*10, (frame_dim, stack_number))
				if big_eval >= min_score:
					print(80 * "=")
					for ev in evaluations:
						print(ev)
					print(f"Solved: At training episode {episode_num} "
											f"mean for {eval_length*10} eps evaluation is "
											f"{np.max(evaluations):.2f} >= {min_score}")
					print(80 * "=")
					break


def eval_policy(policy, env, seed, eval_episodes, stacker_dims, render=True, delay=None):
	rdn = np.random.randint(0, 100)
	env.seed(seed + rdn)

	eval_stacker = FrameStacker(*stacker_dims)

	avg_reward = 0.
	eps = 0
	for _ in range(eval_episodes):
		eps += 1
		frame, done = env.reset(), False
		state = eval_stacker.make_next_state(frame)
		while not done:
			state = np.array(state, dtype='float32')
			action = policy.select_action(state)
			frame, reward, done = env.step(action)
			state = eval_stacker.make_next_state(frame)
			avg_reward += reward
			if render is True:
				env.render('save/')
			if delay is not None:
				time.sleep(delay)
			if done:
				print(40 * "- ")
		eval_stacker.reset()
	avg_reward /= eval_episodes

	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print(40 * "= ")
	return avg_reward


class FrameStacker(object):
	def __init__(self, frame_dim, max_size=int(8)):
		self.frame_dim = frame_dim
		self.max_size = max_size
		self.ptr = 0
		self.first = True
		self.state = np.zeros((self.max_size * self.frame_dim))

	def make_next_state(self, frame):
		if self.first is True:
			for i in range(self.frame_dim):
				self.state = np.roll(self.state, self.frame_dim)
				self.state[:self.frame_dim] = frame
				self.first = False

		else:
			self.state = np.roll(self.state, self.frame_dim)
			self.state[:self.frame_dim] = frame

		return np.array(self.state)

	def reset(self):
		self.state = np.zeros((self.max_size * self.frame_dim))
		self.first = True


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


