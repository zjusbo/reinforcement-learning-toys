'''
	This is the Q-learning function optimizing a grid game: Sarsa on the Windy Gridworld
	The game is proposed in David Silver's reinforcement lecture 5.
	https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=5

	Dyna-Q algorithm:
	   - 

'''
from random import randint, random, choice

class State:
	'''Models each state in the environment. 
	In this particular case, each state is one cell in the grid.
	'''
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __eq__(self, other):
		if isinstance(other, State):
			return self.x == other.x and self.y == other.y
		return False

	def __hash__(self):
		return hash("{},{}".format(self.x, self.y))

STATE_END = State(7, 3)
STATE_INIT = State(0, 3)

class Action:
	'''Models actions agent can take.
	In this case, agent can move in 8 possible directions.
	'''
	def __init__(self, dx, dy):
		self.dx = dx
		self.dy = dy

	def __eq__(self, other):
		if isinstance(other, Action):
			return self.dx == other.dx and self.dy == other.dy
		return False

	def __hash__(self):
		return hash("{},{}".format(self.dx, self.dy))

	def __str__(self):
		if self.dx == 1 and self.dy == 0:
			return 'R'
		if self.dx == -1 and self.dy == 0:
			return 'L'
		if self.dx == 0 and self.dy == 1:
			return 'U'
		if self.dx == 0 and self.dy == -1:
			return 'D'
		if self.dx == 1 and self.dy == 1:
			return 'ru'
		if self.dx == 1 and self.dy == -1:
			return 'rd'
		if self.dx == -1 and self.dy == 1:
			return 'lu'
		if self.dx == -1 and self.dy == -1:
			return 'ld'
		return 'x'

# There are 7 actions we can take in each state (grid). It's a kings move.
ACTIONS = [Action(-1,-1),Action(-1,0),Action(-1,1),Action(0,-1),Action(0,1),Action(1,-1),Action(1,0),Action(1,1)]

class Model:
	'''Model is what agent have learned about the environment
	In this case, it is all the states-actions agent has taken. It stores the reward and the next state given 
	S_{t+1} ~ P(S_t, A_t)
	R_{t+1} = R(S_t, A_t) 
	'''
	def __init__(self):
		self.visited_states = {}

	def visit(self, state: State, action: Action, reward: float, next_state: State):
		'''
		'''
		if state not in self.visited_states:
			self.visited_states[state] = {}
		self.visited_states[state][action] = (next_state, reward)

	def random_pick(self) -> tuple[State, Action]:
		'''Randomly pick one state-action from model memory
		'''
		state = choice(list(self.visited_states.keys()))
		action = choice(list(self.visited_states[state].keys()))
		return state, action

	def get_reward_and_transition_state(self, state: State, action: Action) -> tuple[State, float]:
		'''Given a state and an action, return the reward and transition state by looking up the learned model.
		Returns: next transition state, reward
		'''
		if state not in self.visited_states or action not in self.visited_states[state]:
			raise ValueError('state, action pair does not exist in model')
		return self.visited_states[state][action]


class Environment:
	'''Environment decides how the "wind" blows the agent. The transision function is unknown to the agent. Agent can only observe how envrionment impact the S, a
 	'''
	
	WIND = [0,0,0,1,1,1,2,2,1,0] # Positive numbers mean wind blowing upward
	WIDTH = len(WIND)
	HEIGHT = 7

	@staticmethod
	def transit(state: State, action: Action) -> State:
		'''Given a state and action, return the state the environment will lead us to.'''
		nx = min(max(state.x + action.dx, 0), Environment.WIDTH - 1)
		ny = min(max(state.y + action.dy, 0), Environment.HEIGHT - 1)
		ny = min(ny + Environment.WIND[nx], Environment.HEIGHT - 1)

		return State(nx, ny)

	@staticmethod
	def reward(state: State) -> int:
		'''Returns reward collected when you move out of the State s
		   each state has a -1 reward except the goal state.
		'''
		if state == STATE_END:
			return 100
		return -1	

class StateActionValueFunction:
	def __init__(self, init_value: float, value: dict = None):
		self.init_value = init_value
		self.value = value if value else {}

	def __str__(self):
		width = Environment.WIDTH
		height = Environment.HEIGHT
		output = ''
		for y in reversed(range(height)):
			line = []
			for x in range(width):
				max_q = max(self.get(State(x, y), action) for action in ACTIONS)
				line.append(f'{max_q:.2f}')
			output += "\t".join(line) + '\n'
		return output

	def copy(self):
		return StateActionValueFunction(self.init_value, self.value.copy())

	def get(self, state: State, action: Action) -> float:
		if (state, action) in self.value:
			return self.value[(state, action)]
		else:
			return self.init_value

	def set(self, state: State, action: Action, value: float):
		self.value[(state, action)] = value
	
class QLearning:
	def __init__(self, epsilon: float = 0.1, discount_factor: float = 0.9, learning_rate: float = 0.5):
		self.epsilon = 0 
		self.discount_factor = 0
		self.learning_rate = 0
		self.q_function = StateActionValueFunction(0)
		self.model = Model()
		self.set_epsilon(epsilon).set_discount_factor(discount_factor).set_learning_rate(learning_rate).set_simulation_steps(0).set_max_steps(1000)

	def set_epsilon(self, epsilon: float):
		'''greedy epsilon.
		when set to 0, it is random exploration.
		when set to 1, it is exploitation. Greedy.
		'''
		
		if epsilon > 1 or epsilon < 0:
			raise ValueError(f'epsilon must be between 0 and 1, but got {epsilon}')

		self.epsilon = epsilon
		return self

	def set_discount_factor(self, discount_factor: float):
		'''when set to 0, it does not consider the future value at all (most shortsighted)
		when set to 1, it treats the future value the same as present value (most farsighted)
		'''
		if discount_factor > 1 or discount_factor < 0:
			raise ValueError(f'discount_factor must be between 0 and 1, but got {discount_factor}')

		self.discount_factor = discount_factor
		return self

	def set_learning_rate(self, learning_rate: float):
		if learning_rate > 1 or learning_rate < 0:
			raise ValueError(f'learning_rate must be between 0 and 1, but got {learning_rate}')

		self.learning_rate = learning_rate
		return self

	def set_simulation_steps(self, steps: int):
		self.simulation_steps = steps
		return self

	def set_max_steps(self, steps: int):
		self.max_steps_in_episode = steps
		return self

	def get_next_action(self, state: State) -> Action:
		'''Use epsilon-greedy to choose an action based on the current StateActionValueFunction'''
		best_action = self.__get_best_action(state)
		r = random()
		# with epsilon probability - explore
		if r < self.epsilon:
			return choice(ACTIONS)
		# choose the current best action - exploit
		return best_action

	def __get_best_action(self, state: State) -> Action:
		'''Based on the current StateValueFunction, choose the action that can leads to the max value'''
		return max(ACTIONS, key=lambda a: self.q_function.get(state, a))
	
	def get_lambda_target(self, state: State, action: Action, next_state: State, reward: float) -> float:
		q_max = max(self.q_function.get(next_state, action) for action in ACTIONS)
		lambda_target = reward + self.discount_factor * q_max
		return lambda_target

	def run_simulation(self):
		'''This is the thinking process. Agent simulate movements from their past experience (model they learned)
		'''
		state, action = self.model.random_pick()
		next_state, reward = self.model.get_reward_and_transition_state(state, action)
		lambda_target = self.get_lambda_target(state, action, next_state, reward)
		q = self.q_function.get(state, action)
		q_new = q + self.learning_rate * (lambda_target - q)
		self.q_function.set(state, action, q_new)

	def run_episode(self, state: State) -> tuple[int, float, str]:
		'''
  			Initialize Q(s, a) with arbitrary value, and Q(STATE_END, *) = 0
  			Repeat (for each episode)
  				Initialize S
  				Repeat (for each step of episode)
  					Choose A from S using policy derived from Q (e-greedy)
  					Take action A, observe R, S`
  					Q(S,A) = Q(S,A) + alpha * (R + lambda * max_a Q(S`,a) - Q(S,A))
  					Store S, A, R, S' in model
  					Repeat n times: # This is the thinking/simulation process
  						Sm <- random previously observed state
  						Am <- random previously action taken in Sm
						Sm', Rm <- Model(Sm, Am)
						Q(Sm,Am) = Q(Sm,Am) + alpha * (R + lambda * max_a Q(Sm`,am) - Q(Sm,Am))
  					S = S`
  				Until S is STATE_END
		'''
		steps = 0
		total_reward = 0
		avg_lambda_error = 0
		path = ""
		while state != STATE_END and steps < self.max_steps_in_episode:
			action = self.get_next_action(state)
			path += f'{action} '
			reward = Environment.reward(state)
			next_state = Environment.transit(state, action)
			lambda_target = self.get_lambda_target(state, action, next_state, reward)
			q = self.q_function.get(state, action)
			lambda_error = lambda_target - q
			q_new = q + self.learning_rate * lambda_error

			self.q_function.set(state, action, q_new)

			# update the model
			self.model.visit(state, action, reward, next_state)
			# thinking process
			for i in range(0, self.simulation_steps):
				self.run_simulation()
			state = next_state
			steps += 1
			total_reward += reward
			avg_lambda_error += 1 / steps * (lambda_error - avg_lambda_error)
		total_reward += Environment.reward(state)
		return steps, total_reward, avg_lambda_error, path

	def train(self, episodes: int, verbose = 0):
		steps_list = []
		for _ in range(episodes):
			steps, total_reward, avg_lambda_error, path = self.run_episode(STATE_INIT)
			if steps > 20:
				path = '<path truncated>'
			if verbose > 0:
				print(f"Episode completed in {steps} with total reward {total_reward}. Avgerage lambda error {avg_lambda_error}.  {path}")
			if verbose > 1:
				print(self.q_function)
			steps_list.append(steps)
		return steps_list

def plot(values: list, title: str = ''):
	import numpy as np
	import matplotlib.pyplot as plt

	x = [x for x in range(0, len(values[0][0]))]
	
	for value, label in values:
		# Plot each piece
		plt.plot(x, value, label=label)
	
	# Add titles and labels
	plt.xlabel("episode count")
	plt.ylabel("steps per episode")
	plt.title(title)
	plt.legend()

	# Show the plot
	plt.show()


if __name__ == "__main__":
	result = []
	# search learning rate
	for learning_rate in [0.1, 0.3, 0.6, 1]:
		agent = QLearning()
		agent.set_learning_rate(learning_rate)
		steps = agent.train(episodes=50)
		result.append((steps, f'learning rate = {learning_rate}'))

	plot(result,title="steps per episode under different learning rates" )
	# the best learning rate is 1.
	learning_rate = 1
	
	# search discount_factor
	result = []
	for discount_factor in [0, 0.1, 0.3, 0.6, 0.9, 1]:
		agent = QLearning()
		agent.set_learning_rate(learning_rate).set_discount_factor(discount_factor)
		steps = agent.train(episodes=50)
		result.append((steps, f'discount factor = {discount_factor}'))

	plot(result, title=f"steps per episode at learning rate = {learning_rate}")
	# We did not see much performance difference between any positive discount factors.
	discount_factor = 0.9

	# search epsilon
	result = []
	for epsilon in [0, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]:
		agent = QLearning()
		agent.set_learning_rate(learning_rate).set_discount_factor(discount_factor).set_epsilon(epsilon)
		steps = agent.train(episodes=50)
		result.append((steps, f'epsilon = {epsilon}'))

	plot(result, title=f"steps per episode at learning rate = {learning_rate}, discount_factor = {discount_factor}")
	# The bset epsilon is between 0 and 0.1
	epsilon = 0.1

	# search simulation steps
	result = []
	for simulation_steps in [0, 3, 6, 10, 30, 60, 100]:
		agent = QLearning()
		agent.set_learning_rate(learning_rate).set_discount_factor(discount_factor).set_epsilon(epsilon).set_simulation_steps(simulation_steps)
		steps = agent.train(episodes=20)
		result.append((steps, f'simulation_steps = {simulation_steps}'))

	plot(result, title=f"steps per episode at learning rate = {learning_rate}, discount_factor = {discount_factor}, epsilon = {epsilon}")
	# the best simulation steps is 60
	simulation_steps = 60

	result = []
	agent = QLearning()
	agent.set_learning_rate(learning_rate).set_discount_factor(discount_factor).set_epsilon(epsilon).set_simulation_steps(simulation_steps)
	steps = agent.train(episodes=20)
	result.append((steps, f''))
	plot(result, title=f"steps per episode at learning rate = {learning_rate}, discount_factor = {discount_factor}, epsilon = {epsilon}, simulation_steps = {simulation_steps}")

	# On average, it takes 2 episodes to find the optimal path.
