import numpy as np
import time

class maze_world():
    def __init__(self, map_size, start_point, goal_point, walls):
        self.maze_env = self.initialize(map_size, walls) 
        self.start_point = start_point.copy()
        self.goal_point = goal_point.copy()
        self.walls = walls.copy()
        self.cur_state = start_point.copy()
        self.map_size = map_size.copy()
        
        self.up = 0
        self.down = 1
        self.left = 2
        self.right = 3

    def initialize(self, map_size, walls):
        maze_env = np.zeros((map_size[0], map_size[1]))
        for w in walls:
            maze_env[w[0]][w[1]] = 1
        return maze_env

    def state_0(self):
        self.cur_state = self.start_point.copy()
        return self.start_point

    def step(self, action):

        reward = 0
        done = True
        if action == self.up: 
            self.cur_state[0] -= 1
        elif action == self.down:
            self.cur_state[0] += 1
        elif action == self.left:
            self.cur_state[1] -= 1
        else:
            self.cur_state[1] += 1

        if self.cur_state[0] == self.goal_point[0] and self.cur_state[1] == self.goal_point[1]:
            reward = 1
            done = False
        output = self.cur_state.copy()
    
        return output, reward, done

    def check_available_action(self, state, action):
        temp_state = state.copy()
        
        if action == self.up:
            temp_state[0] -= 1
        elif action == self.down:
            temp_state[0] += 1
        elif action == self.left:
            temp_state[1] -= 1
        else:
            temp_state[1] += 1

        if [temp_state[0], temp_state[1]] in self.walls:
            return False

        if temp_state[0] >= 0 and temp_state[0] < self.map_size[0] and temp_state[1] >= 0 and temp_state[1] < self.map_size[1]:
            return True
        return False
    
    def render(self):
        return 0

class Dyna_q():
    def __init__(self, epsilon, alpha, action_size, map_size, world):
        self.world_model = []
        self.Q = self.initialize(action_size, map_size, world)
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_size = action_size

    def initialize(self, action_size, map_size, world):
        Q = np.zeros((action_size, map_size[0], map_size[1]))
        return Q


    def epsilon_greedy(self, world, state):
        if np.random.binomial(1, self.epsilon) == 1: # epsilon의 확률로 1이 선택된다. 90%는 0, 10%는 1 즉, 10%의 확률로 random하게 action을 선택한다.
            while True:
                action = np.random.randint(self.action_size)
                if world.check_available_action(state, action):
                    return action
        else:
            Q_values = self.Q[:,state[0], state[1]]
            max_Q_value = np.max(Q_values)
            while True:
                action = np.random.choice([action for action, Q_values in enumerate(Q_values) if Q_values == max_Q_value]) # 같은 Q_value를 가진 action을 모두 찾은 후 random 선택.
                if world.check_available_action(state, action):
                    return action

    def Q_update(self, state, action, next_state, reward):
        self.Q[action, state[0], state[1]] += self.alpha * (reward + np.max(self.Q[:,next_state[0], next_state[1]]) - self.Q[action, state[0], state[1]])
      
    def learn_world_model(self, state, action, next_state, reward):
        self.world_model.append((state, action, next_state, reward))
    
    def random_s_a(self):
        len_data = len(self.world_model)
        idx = np.random.randint(len_data)
        return self.world_model[idx][0], self.world_model[idx][1]

    def forward_world_model(self, state, action):
        world_model = np.array(self.world_model)
        
        idx = np.random.choice([index for index, data in enumerate(world_model) if data[0][0] == state[0] and  data[0][1] == state[1] and data[1] == action])
        output = self.world_model[idx]
        reward = output[3]
        next_state = output[2]
        return reward, next_state
    
    def optimal_policy(self):
        

def main():
    epsilon = 0.1
    alpha = 0.5
    action_size = 4
    map_size = np.array([6,9])
    start_point = np.array([2,0])
    goal_point = np.array([0,8])
    walls = [[1,2],[2,2],[3,2],[4,5],[0,7],[1,7],[2,7]]

    episode = 2
    N_times = 50

    world = maze_world(map_size, start_point, goal_point, walls)
    print(world.maze_env)
    Dyna = Dyna_q(epsilon, alpha, action_size, map_size, world)
    print(Dyna.Q)
    for epi in range(episode):
        done = True
        state = world.state_0()
        while done:
            action = Dyna.epsilon_greedy(world, state)
            next_state, reward, done = world.step(action)
            Dyna.Q_update(state, action, next_state, reward)
            Dyna.learn_world_model(state, action, next_state, reward)
            state = next_state.copy()
            for n in range(N_times):
                h_state, h_action = Dyna.random_s_a()
                h_reward, h_next_state = Dyna.forward_world_model(h_state, h_action)
                Dyna.Q_update(h_state, h_action, h_next_state, h_reward)
    

    print(Dyna.Q)
if __name__ == '__main__':
    main()