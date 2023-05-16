import heapq
import copy
from time import perf_counter

class Problem:
  def __init__(self, initial_state, goal_state, algo=1):  
    self.initial_state = initial_state
    self.goal_state = goal_state
    self.algo = algo
    self.operators = ["move_down", "move_left", "move_up", "move_right"]
  
  # move tile up
  def move_up(self, state,i,j):
    n = len(state)
    if (i != (n - 1)):
      state[i][j], state[i + 1][j] = state[i + 1][j], 0
      return state
    return None
  
  # move tile down
  def move_down(self, state,i,j):
    if (i != 0):
      state[i][j], state[i - 1][j] = state[i - 1][j], 0
      return state
    return None

  # move tile left
  def move_left(self, state,i,j):
    n = len(state)
    if (j != (n - 1)):
      state[i][j],  state[i][j + 1] = state[i][j + 1], 0
      return state
    return None

  # move tile right
  def move_right(self, state,i,j):
    if (j != 0):
      state[i][j], state[i][j - 1] = state[i][j - 1], 0
      return state
    return None


# Class to define node
class Node:
  def __init__(self, parent=None, gn=0, hn=0, fn=0, state=0):  
    self.children = []
    self.parent = parent
    self.state = state
    self.gn = 0
    self.hn = 0
    self.fn = 0

  # pop the state from the heapq whose f(n) is minimum (i.e less steps to reach the goal)
  def __lt__(self, node):
      return self.fn < node.fn

  def __eq__(self, node):
      return self.state == node.state

  # add child to the node
  def add_child(self, node, cost_to_expand=1):
        node.gn = self.gn + cost_to_expand
        node.fn = node.gn + node.hn
        node.parent = self
        self.children.append(node)

  # function to expand a node
  def expand(self, problem):
        # find index of 0
        i, j = find_index(0, self.state)
        states = []
        
        # expand state with all possible moves
        for o in problem.operators:
          state = copy.deepcopy(self.state)
          state = getattr(problem, o)(state,i,j)
          
          if state:
            # Don't expand parent state again
            if self.parent and compare_states(state, self.parent.state):
                states.append(None) 
            else:
                states.append(state)
               
        return states


# Compare 2 states
def compare_states(state1, state2):
    for i in range(len(state1)):
      if state1[i] != state2[i]:
        return False
    return True
  
# Find index of the element in the 2d array
def find_index(element, state):
  for i, row in enumerate(state):
      try:
          j = row.index(element)
      except ValueError:
          continue
      return i, j
          
  return None, None

# calculate misplaced tiles
def misplaced_tiles(state, goal):
    tiles = 0
    for i in range(len(goal)):
      for j in range(len(goal[0])):
          if state[i][j] == 0:
              continue
          if (state[i][j] != goal[i][j]):
              tiles += 1 
    return tiles

# calculate manhattan distance
def manhattan_distance(state, goal):
    dist = []

    for i in range(0, len(goal)):
        for j in range(0, len(goal[i])):
            if (state[i][j] == goal[i][j]):
                continue
                
            if (state[i][j] == 0):
                continue
            else:
                i_goal, j_goal = find_index(state[i][j], goal)
                distance = abs(i - i_goal) + abs(j - j_goal)
                dist.append(distance)
    
    return sum(dist)


# general search function for exploring all the states until we get goal state
def general_search(problem, start_time, max_time):
  root = Node(state=problem.initial_state)
  nodes = []
  heapq.heappush(nodes, root)
  visited = []
  max_nodes = 1
  expanded_nodes = 0
  while nodes and (perf_counter() - start_time) < max_time:
    max_nodes = max(len(nodes), max_nodes)
    curr = heapq.heappop(nodes)
    if (compare_states(curr.state, problem.goal_state)):
      print("\nPuzzle Solved!")
      print("Depth of the solution: ", curr.gn)
      print("Total expanded nodes: ", expanded_nodes) 
      print("The maximum number of nodes in the queue at any time: " + str(max_nodes))
      total_time = (perf_counter() - start_time)
      print("Total time required in milliseconds: ", round(total_time*1000, 2))
              
      return expanded_nodes, max_nodes
    else:
      visited.append(curr)
      expanded_states = [i for i in curr.expand(problem) if i is not None]
      if not expanded_states:
        continue
      for s in expanded_states:
        node = Node(state=s)

        if ((nodes and node in nodes) or (visited and node in visited)):
            continue

        # Depending on the heuristic, calculate h(n)
        if (problem.algo == 2):
            node.hn = misplaced_tiles(node.state, problem.goal_state)

        if (problem.algo == 3):
            node.hn = manhattan_distance(node.state, problem.goal_state)

        curr.add_child(node=node)
        heapq.heappush(nodes, node)

    expanded_nodes += 1

  if nodes:
    print("Time limit exceeded... Could not find a solution within given time limit.")
  else:
    print("Couldn't find a solution...")            


print("Please enter the puzzle dimension n: ")
n = int(input())

print("Enter elements of the initial state of {n}*{n} puzzle row wise (enter 0 for empty cell)".format(n=n))
initial_state = []
for i in range(n):
  initial_state.append(list(map(int,input().split())))

print("\nThe initial state is:")
for i in range(n):
  for j in range(n):
    print(initial_state[i][j], end=" ")
  print()

print("\nEnter elements of the goal state of {n}*{n} puzzle row wise (enter 0 for empty cell)".format(n=n))
goal_state = []
for i in range(n):
  goal_state.append(list(map(int,input().split())))
  
print("\nThe goal state is:")
for i in range(n):
  for j in range(n):
    print(goal_state[i][j], end=" ")
  print()
  

while True:
  print("\nPlease select one of the algorithms to solve the problem\n 1. Uniform Cost Search\n 2. A* Misplaced Tile Heuristic\n 3. A* Manhattan Distance Heuristic")
  algo = int(input())
  if algo == 1:
    print("Uniform search will be used to solve this puzzle")
    break
  elif algo == 2:
    print("Misplaced tile heuristic will be used to solve this puzzle")
    break
  elif algo == 3:
    print("Manhattan distance heuristic will be used to solve this puzzle")
    break
  else:
    print("Please enter valid choice")

print("\nDo you want to set a time limit for the puzzle? y/n ")
choice = input()
if choice.lower() == "y":
  print("Please enter the time limit in seconds")
  max_time = int(input())
else:
  print("Default time limit of 600 seconds (10 minutes) is set")
  max_time = 600

start_time = perf_counter()
problem = Problem(initial_state, goal_state, algo)
general_search(problem, start_time, max_time)
