import numpy as np
import cvxpy as cp
import json
import os

TEAM_NUMBER = 22
ARR = [1/2, 1, 2]
Y = ARR[TEAM_NUMBER % 3]
STEP_COST = -10 / Y

# STATE VALUES
POS_W = 0
POS_N = 1
POS_E = 2
POS_S = 3
POS_C = 4
POS_STRING = ['W', 'N', 'E', 'S', 'C']
MAX_POSITIONS = 5
MM_HEALTHS = [0, 25, 50, 75, 100]
NUM_MM_HEALTHS = 5
MAX_MM_HEALTH = 100
MAX_ARROWS = 3
MAX_MATERIALS = 2
MM_READY = 1
MM_DORMANT = 0
MAX_MM_STATES = 2
MM_STATES_STRING = ['D', 'R']

# (<pos>,<mat>,<arrow>,<state>,<health>):<action>=[<state_value>]”
NUM_STATES = MAX_POSITIONS * (MAX_MATERIALS + 1) * (MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS)
HIT_BY_MM_REWARD = -40

# ACTIONS
MOVEMENT_ACTIONS = np.array(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])
ACTIONS = {
    POS_C: {
        "UP": {
            "S": (0.85, POS_N),
            "F": (0.15, POS_E)
        },
        "DOWN": {
            "S": (0.85, POS_S),
            "F": (0.15, POS_E)
        },
        "LEFT": {
            "S": (0.85, POS_W),
            "F": (0.15, POS_E)
        },
        "RIGHT": {
            "S": (0.85, POS_E),
            "F": (0.15, POS_E)
        },
        "STAY": {
            "S": (0.85, POS_C),
            "F": (0.15, POS_E)
        },
        "SHOOT": {
            "S": 0.5,
            "F": 0.5
        },
        "HIT": {
            "S": 0.1,
            "F": 0.9
        }
    },
    POS_N: {
        "DOWN": {
            "S": (0.85, POS_C),
            "F": (0.15, POS_E)
        },
        "STAY": {
            "S": (0.85, POS_N),
            "F": (0.15, POS_E)
        },
        "CRAFT": {
            "1": 0.5,
            "2": 0.35,
            "3": 0.15
        }
    },
    POS_S: {
        "UP": {
            "S": (0.85, POS_C),
            "F": (0.15, POS_E)
        },
        "STAY": {
            "S": (0.85, POS_S),
            "F": (0.15, POS_E)
        },
        "GATHER": {
            "S": 0.75,
            "F": 0.25
        }
    },
    POS_E: {
        "STAY": {
            "S": (1, POS_E),
            "F": (0, POS_E)
        },
        "LEFT": {
            "S": (1, POS_C),
            "F": (0, POS_E)
        },
        "SHOOT": {
            "S": 0.9,
            "F": 0.1
        },
        "HIT": {
            "S": 0.2,
            "F": 0.8
        }
    },
    POS_W: {
        "STAY": {
            "S": (1, POS_W),
            "F": (0, POS_E)
        },
        "RIGHT": {
            "S": (1, POS_C),
            "F": (0, POS_E)
        },
        "SHOOT": {
            "S": 0.25,
            "F": 0.75
        }
    }
}
MM_ACTIONS = {
    "DORMANT": {
        "STAY": 0.8,
        "GET_READY": 0.2
    },
    "READY": {
        "ATTACK": 0.5,
        "STAY": 0.5
    }
}

class State:
    def __init__(self, position_ij, num_materials, num_arrows, state_mm, health_mm):
        self.position_ij = position_ij
        self.health_mm = health_mm
        self.num_arrows = num_arrows
        self.num_materials = num_materials
        self.state_mm = state_mm

    def get_state(self):
        return self.position_ij, self.num_materials, self.num_arrows, self.state_mm, self.health_mm

    def get_position(self):
        return self.position_ij

    def change_position(self, position_ij):
        self.position_ij = position_ij

    def get_arrows(self):
        return self.num_arrows

    def change_arrows(self, arrows):
        self.num_arrows = min(MAX_ARROWS, arrows)
        self.num_arrows = max(0, self.num_arrows)

    def reduce_mm_health(self, val):
        self.health_mm -= val
        self.health_mm = max(0, self.health_mm)
        self.health_mm = min(NUM_MM_HEALTHS - 1, self.health_mm)

    def increase_mm_health(self, val):
        self.health_mm += val
        self.health_mm = max(0, self.health_mm)
        self.health_mm = min(NUM_MM_HEALTHS - 1, self.health_mm)

    def get_materials(self):
        return self.num_materials

    def change_materials(self, count):
        self.num_materials = count
        self.num_materials = min(MAX_MATERIALS, self.num_materials)
        self.num_materials = max(0, self.num_materials)

    def get_mm_state(self):
        return self.state_mm

    def set_mm_ready(self):
        self.state_mm = MM_READY
    
    def set_mm_dormant(self):
        self.state_mm = MM_DORMANT

    def get_index(self):
        return (
            self.position_ij * ((MAX_MATERIALS + 1) * (MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS)) +
            self.num_materials * ((MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS)) +
            self.num_arrows * (MAX_MM_STATES * (NUM_MM_HEALTHS)) +
            self.state_mm * (NUM_MM_HEALTHS) + 
            self.health_mm
        )

    # returns {'action' : [( probability, next_state, got_hurt)] }
    def actions(self):
        if self.health_mm == 0:
            return None

        actions_dic = {}
        for action, result in ACTIONS[self.position_ij].items():
            new_state = State(*self.get_state())
            responses = []
            if action in MOVEMENT_ACTIONS:
                for _, (p, pos) in result.items():
                    new_state.change_position(pos)    
                    responses.append((p, new_state.get_state()))

                actions_dic[action] = (self.mm_effect(action, responses))

            elif action == "SHOOT":
                if self.num_arrows == 0:
                    continue

                # Failure
                new_state.change_arrows(self.num_arrows - 1)
                responses.append((result['F'], new_state.get_state()))
                # Success
                new_state.reduce_mm_health(1)
                responses.append((result['S'], new_state.get_state()))
                actions_dic[action] = (self.mm_effect(action, responses))

            elif action == "HIT":
                if action not in actions_dic:
                    actions_dic[action] = []
                # Failure
                responses.append((result['F'], new_state.get_state()))
                # Success
                new_state.reduce_mm_health(2)
                responses.append((result['S'], new_state.get_state()))
                actions_dic[action] = (self.mm_effect(action, responses))

            elif action == "CRAFT":
                if self.num_materials == 0:
                    continue

                new_state.change_materials(new_state.get_materials() - 1)
                for count, p in result.items():
                    # print(count, p)
                    count = int(count)
                    assert count >= 0
                    new_state.change_arrows(new_state.get_arrows() + count)
                    responses.append((p, new_state.get_state()))
                    new_state.change_arrows(new_state.get_arrows() - count)
                
                actions_dic[action] = (self.mm_effect(action, responses))

            elif action == "GATHER":
                # Failure
                responses.append((result['F'], new_state.get_state()))
                # Success
                new_state.change_materials(new_state.get_materials() + 1)
                responses.append((result['S'], new_state.get_state()))

                actions_dic[action] = (self.mm_effect(action, responses))

        new_actions_dic = {}
        for action in actions_dic.keys():
            temp = []
            for result in actions_dic[action]:
                if result[1] != self.get_state() and result[0] != 0:
                    temp.append(result)

            if len(temp) > 0:
                new_actions_dic[action] = temp

        return new_actions_dic

    # (p, state_tup) => [(p, next_state, got_hurt)]
    def mm_effect(self, action, responses):

        final_responses = []
        if self.state_mm == MM_DORMANT:
            for p, next_state in responses:
                next_state = State(*next_state)
                final_responses.append((p * MM_ACTIONS["DORMANT"]["STAY"], next_state.get_state(), 0))
                next_state.set_mm_ready()
                final_responses.append((p * MM_ACTIONS["DORMANT"]["GET_READY"], next_state.get_state(), 0))

        elif self.state_mm == MM_READY:
            for p, next_state in responses:
                # STAY
                next_state = State(*next_state)
                final_responses.append((p * MM_ACTIONS["READY"]["STAY"], next_state.get_state(), 0))
                # ATTACK
                if self.position_ij != POS_C and self.position_ij != POS_E:
                    next_state.set_mm_dormant()
                    final_responses.append((p * MM_ACTIONS["READY"]["ATTACK"], next_state.get_state(), 0))
         
            if self.position_ij == POS_C or self.position_ij == POS_E:
                next_state = State(*self.get_state())
                next_state.change_arrows(0)
                next_state.increase_mm_health(1)
                next_state.set_mm_dormant()
                final_responses.append((MM_ACTIONS["READY"]["ATTACK"], next_state.get_state(), 1))
        
        return final_responses
            
    @classmethod
    def from_index(self, idx):
        position_ij = idx // ((MAX_MATERIALS + 1) * (MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS))
        idx = idx % ((MAX_MATERIALS + 1) * (MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS))

        num_materials = idx // ((MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS)) 
        idx = idx % ((MAX_ARROWS + 1) * MAX_MM_STATES * (NUM_MM_HEALTHS)) 

        num_arrows = idx // (MAX_MM_STATES * (NUM_MM_HEALTHS)) 
        idx = idx % (MAX_MM_STATES * (NUM_MM_HEALTHS)) 

        state_mm = idx // (NUM_MM_HEALTHS)  
        idx = idx % (NUM_MM_HEALTHS)  
        
        health_mm = idx

        s = State(position_ij, num_materials, num_arrows, state_mm, health_mm)
        return s

def convert_state(state):
    pos_names = ['W', 'N', 'E', 'S', 'C']
    mm_state_name = ['D', 'R']
    pos = pos_names[state[0]]
    mm_state = mm_state_name[state[3]]
    return pos, state[1], state[2], mm_state, state[4]

def print_actions(state, actions):
    state = convert_state(state)
    for action, results in actions.items():
        print(state, action)
        for result in results:
            print(result)

class LinearProgramming:
    def __init__(self):
        self.action_at_idx = [] 
        self.dim = self.get_dimensions()
        self.set_r()
        self.set_a()
        self.set_alpha()
        self.objective = 0.0
        self.x = self.start_lp()
        self.policy = []
        self.output_dict = {}
    
    def get_dimensions(self):
        dim = 0
        for i in range(NUM_STATES):
            if State.from_index(i).actions() is None:
                dim += 1
            else:
                dim += len(State.from_index(i).actions().keys())
        return dim

    def set_a(self):
        # dim = number of (state, action)
        self.a = np.zeros((NUM_STATES, self.dim), dtype=np.float64)
        cnt = 0
        for i in range(NUM_STATES):
            state = State.from_index(i)
            actions = state.actions()

            if actions is None:
                self.action_at_idx.append("NONE")
                self.a[i][cnt] += 1
                cnt += 1
                # print(convert_state(State.from_index(i).get_state()), "NONE")
                continue
            
            # print_actions(State.from_index(i).get_state(), actions)
            
            for action, results in actions.items():
                self.action_at_idx.append(action)
                for result in results:
                    p = result[0]
                    next_state = result[1]
                    self.a[i][cnt] += p
                    self.a[State(*next_state).get_index()][cnt] -= p

                try:
                    assert self.a[i][cnt] <= 1.1
                except:
                    print(State.from_index(i).get_state())
                    print(actions, action, results)
                    assert False
                    
                # increment cnt
                cnt += 1
        
        assert cnt == self.dim 

    def set_r(self):
        self.r = np.zeros((1, self.dim))
        idx = 0
        for i in range(NUM_STATES):
            actions = State.from_index(i).actions()
            
            if actions is None:
                self.r[0][idx] = 0
                idx += 1
                continue

            for _, results in actions.items():
                self.r[0][idx] += (STEP_COST)
                for result in results:
                    p = result[0]
                    got_hurt = result[2]
                    if got_hurt: 
                        self.r[0][idx] += (p * HIT_BY_MM_REWARD)
                idx += 1

    def set_alpha(self):
        # equal probability for start states
        self.alpha = np.zeros((NUM_STATES, 1)) 
        s1 = State(POS_C, 2, 3, MM_READY, 4)
        self.alpha[s1.get_index()] = 1

    def start_lp(self):
        x = cp.Variable((self.dim, 1), 'x')

        constraints = [
            cp.matmul(self.a, x) == self.alpha,
            x >= 0
        ]

        objective = cp.Maximize(cp.matmul(self.r, x))
        problem = cp.Problem(objective, constraints)

        solution = problem.solve(verbose = True)
        self.objective = solution
        
        l = [ float(val) for val in list(x.value)]
        return l

    def find_policy(self):
        idx = 0
        for i in range(NUM_STATES):
                
            state = list(convert_state(State.from_index(i).get_state()))
            actions = State.from_index(i).actions()
            state[4] *= 25

            if actions is None: 
                idx += 1
                best_action = "NONE"
                self.policy.append([state, best_action])
                continue

            act_idx = np.argmax(self.x[idx : idx+len(actions.keys())])
            idx += len(actions.keys())
            best_action = list(actions.keys())[act_idx]
            self.policy.append([state, best_action])
 

    def output(self):
        self.output_dict["a"] = self.a.tolist()
        self.output_dict["r"] = [float(val) for val in np.transpose(self.r)]
        self.output_dict["alpha"] = [float(val) for val in self.alpha]
        self.output_dict["x"] = self.x
        self.output_dict["policy"] = self.policy
        self.output_dict["objective"] = float(self.objective)
        
        os.makedirs('outputs', exist_ok=True)
        path = "outputs/part_3_output.json"
        with open(path, 'w') as file:
            json.dump(self.output_dict, file, indent=4)

    def run(self):
        self.find_policy()
        self.output()

if __name__ == '__main__':
    lp = LinearProgramming()
    lp.run()
