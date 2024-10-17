# Lecture 3 - verify the expected steps for the 4 * 4 grid
import random


class Character:
	directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.x_max = 3
		self.y_max = 3
	def move(self, v):
		v = v % 4
		direction = Character.directions[v]
		nx = self.x + direction[0]
		ny = self.y + direction[1]
		nx = max(0, min(nx, self.x_max))
		ny = max(0, min(ny, self.y_max))
		self.x = nx
		self.y = ny
	def get_position(self):
		return (self.x, self.y)
	def is_end(self):
		if self.x == 0 and self.y == 0:
			return True
		if self.x == self.x_max and self.y == self.y_max:
			return True
		return False

avg_count = 0
iteration = 10000
for i in range(0, iteration):
	c = Character(0, 1)	
	step_count = 0
	while not c.is_end():
	  v = random.randint(0, 3)
	  c.move(v)
	  step_count += 1
	avg_count += step_count
avg_count = avg_count / iteration
print("steps {}".format(avg_count))
