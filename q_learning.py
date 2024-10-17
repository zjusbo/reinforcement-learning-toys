'''
	This is the Q-learning function optimizing a grid game: Sarsa on the Windy Gridworld
	The game is proposed in David Silver's reinforcement lecture 5.
	https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=5
'''
from random import randint, random, choice

class State:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __eq__(self, other):
		if isinstance(other, State):
			return self.x == other.x and self.y == other.y
		return False

	def __hash__(self):
		return hash("{},{}".format(self.x, self.y))

	def reward(self) -> int:
		'''Returns reward collected when you move out of the State s
		   each state has a -1 reward except the goal state.
		'''
		if self == STATE_END:
			return 100
		return -1	

STATE_END = State(7, 3)
STATE_INIT = State(0, 3)

class Action:
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

class Environment:
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
	def __init__(self, epsilon: float, discount_factor: float, learning_rate: float):
		if epsilon >= 1 or epsilon < 0:
			raise ValueError(f'epsilon must be between 0 and 1, but got {epsilon}')

		if discount_factor >= 1 or discount_factor < 0:
			raise ValueError(f'discount_factor must be between 0 and 1, but got {discount_factor}')

		if learning_rate >= 1 or learning_rate < 0:
			raise ValueError(f'learning_rate must be between 0 and 1, but got {learning_rate}')
		
		self.epsilon = epsilon
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.q_function = StateActionValueFunction(0)

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
	
	def run_episode(self, state: State, max_steps: int = 1000) -> tuple[int, float, str]:
		'''
  			Initialize Q(s, a) with arbitrary value, and Q(STATE_END, *) = 0
  			Repeat (for each episode)
  				Initialize S
  				Repeat (for each step of episode)
  					Choose A from S using policy derived from Q (e-greedy)
  					Take action A, observe R, S`
  					Q(S,A) = Q(S,A) + alpha * (R + lambda * max_a Q(S`,a) - Q(S,A))
  					S = S`
  				Until S is STATE_END
		'''
		steps = 0
		total_reward = 0
		avg_lambda_error = 0
		path = ""
		while state != STATE_END and steps < max_steps:
			action = self.get_next_action(state)
			path += f'{action} '
			reward = state.reward()
			next_state = Environment.transit(state, action)

			q_max = max(self.q_function.get(next_state, action) for action in ACTIONS)
			q = self.q_function.get(state, action)
			lambda_error = reward + self.discount_factor * q_max - q
			q_new = q + self.learning_rate * lambda_error
			
			self.q_function.set(state, action, q_new)
			state = next_state
			steps += 1
			total_reward += reward
			avg_lambda_error += 1 / steps * (lambda_error - avg_lambda_error)
		total_reward += state.reward()
		return steps, total_reward, avg_lambda_error, path

	def train(self, episodes: int):
		for _ in range(episodes):
			steps, total_reward, avg_lambda_error, path = self.run_episode(STATE_INIT)
			if steps > 20:
				path = ''
			print(f"Episode completed in {steps} with total reward {total_reward}. Avgerage lambda error {avg_lambda_error}.  {path}")
			print(self.q_function)

if __name__ == "__main__":
    agent = QLearning(epsilon=0.1, discount_factor=0.9, learning_rate=0.5)
    agent.train(episodes=300)
