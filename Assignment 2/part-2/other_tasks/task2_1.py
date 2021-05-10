import numpy as np
from copy import deepcopy
import sys
import json

# HYPERPARAMTERS
# GAMMA = 0.25
GAMMA = 0.999
BELLMON_ERROR = 0.001
TEAM_NUMBER = 22
ARR = [1/2, 1, 2]
Y = ARR[TEAM_NUMBER % 3]
STEP_COST = -10 / Y
# STEP_COST = -20

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

# REWARDS
GAME_PRIZE = 50
# (<pos>,<mat>,<arrow>,<state>,<health>):<action>=[<state_value>]”
STATE_REWARD = np.zeros((MAX_POSITIONS, MAX_MATERIALS + 1, MAX_ARROWS + 1, MAX_MM_STATES, MAX_MM_HEALTH + 1))
                         
STATE_REWARD[:, :, :, :, 0] = GAME_PRIZE
HIT_BY_MM_REWARD = -40

MM_NOT_ATTACKED = 0
MM_UNSUCCESSFUL_ATTACK = 1
MM_SUCCESSFULL_ATTACK = 2

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
            "S": (1, POS_W),
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
        self.health_mm = min(MAX_MM_HEALTH, self.health_mm)

    def increase_mm_health(self, val):
        self.health_mm += val
        self.health_mm = max(0, self.health_mm)
        self.health_mm = min(MAX_MM_HEALTH, self.health_mm)

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


# (State, V) => all_action_results [(p1, next_state1).....]
def action(state_tup, V):
    all_action_results = {}
    state = State(*state_tup)
    # Finding Best Action
    for action, result in ACTIONS[state.get_position()].items():
        new_state = State(*state_tup)
        action_results = []
        if action in MOVEMENT_ACTIONS:
            for _, (p, pos) in result.items():
                new_state.change_position(pos)
                action_results.append((p, new_state.get_state()))

        elif action == "SHOOT":
            if state.get_arrows() == 0:
                continue
            
            new_state.change_arrows(state.get_arrows() - 1)
            # Failure
            action_results.append((result['F'], new_state.get_state()))
            # Success
            new_state.reduce_mm_health(25)
            action_results.append((result['S'], new_state.get_state()))

        elif action == "HIT":
            # Failure
            action_results.append((result['F'], new_state.get_state()))
            # Success
            new_state.reduce_mm_health(50)
            action_results.append((result['S'], new_state.get_state()))

        elif action == "CRAFT":
            if state.get_materials() == 0:
                continue

            new_state.change_materials(new_state.get_materials() - 1)
            for count, p in result.items():
                # print(count, p)
                count = int(count)
                assert count >= 0
                new_state.change_arrows(new_state.get_arrows() + count)
                action_results.append((p, new_state.get_state()))
                new_state.change_arrows(new_state.get_arrows() - count)

        elif action == "GATHER":
            # Failure
            action_results.append((result['F'], new_state.get_state()))
            # Success
            new_state.change_materials(new_state.get_materials() + 1)
            action_results.append((result['S'], new_state.get_state()))

        all_action_results[action] = action_results

    if not all_action_results:
        return None
    return all_action_results

# (state, V, responses) => taken, new_res
def mm_action(state_tup, V, all_action_results):
    best_action = None 
    best_action_responses = None
    all_action_responses = {}

    max_value = np.NINF
    for action, responses in all_action_results.items():
        state = State(*state_tup)
        value = 0
        final_responses = []
        if state.get_mm_state() == MM_DORMANT:
            for p, next_state in responses:
                next_state = State(*next_state)
                final_responses.append((p * MM_ACTIONS["DORMANT"]["STAY"], next_state.get_state(), V[next_state.get_state()], MM_NOT_ATTACKED))
                next_state.set_mm_ready()
                final_responses.append((p * MM_ACTIONS["DORMANT"]["GET_READY"], next_state.get_state(), V[next_state.get_state()], MM_NOT_ATTACKED))

        elif state.get_mm_state() == MM_READY:
            for p, next_state in responses:
                # STAY
                next_state = State(*next_state)
                final_responses.append((p * MM_ACTIONS["READY"]["STAY"], next_state.get_state(),  V[next_state.get_state()], MM_NOT_ATTACKED))
                # ATTACK
                if state.get_position() != POS_C and state.get_position() != POS_E:
                    next_state.set_mm_dormant()
                    final_responses.append((p * MM_ACTIONS["READY"]["ATTACK"], next_state.get_state(),  V[next_state.get_state()], MM_UNSUCCESSFUL_ATTACK))

            if state.get_position() == POS_C or state.get_position() == POS_E:
                state.change_arrows(0)
                state.increase_mm_health(25)
                state.set_mm_dormant()
                final_responses.append((MM_ACTIONS["READY"]["ATTACK"], state.get_state(),  V[state.get_state()], MM_SUCCESSFULL_ATTACK))
        

        for (p, s1, _, got_hurt) in final_responses:
            if got_hurt == MM_SUCCESSFULL_ATTACK:
                value += (p * HIT_BY_MM_REWARD)
                
            value += ( p * (STEP_COST + STATE_REWARD[s1] + GAMMA * (V[s1])))

        if value >= max_value:
            best_action = action
            best_action_responses = final_responses
            max_value = value

        all_action_responses[action] = final_responses
    
    # for action, responses in all_action_responses.items():
    #     print(action)
    #     for now in responses:
    #         print(now)
    
    return best_action, best_action_responses, max_value, all_action_responses

def human_readable_state(state):
    position_names = ['W', 'N', 'E', 'S', 'C']
    mm_state_names = ['D', 'R']
    temp = list(state)
    temp[0] = position_names[temp[0]]
    temp[3]= mm_state_names[temp[3]]
    return tuple(temp)

# V [state => value], actions[state : action], results[state: best_action_responses]
def Simulate(V, actions, results, start):
    f = open("Simulation.txt", "w")
    f.write("Simulation from {}\n".format(human_readable_state(start)))
    state = start
    it = 0
    while it < 100:
        if state[-1] == 0:
            break
        s = ""
        
        s += 'CURR: {}  '.format(human_readable_state(state))
        p = np.random.rand()
        # s += 'P: {} '.format(p)
        sum_p = 0
        next_state = None
        chosen_action = None
        attack_by_mm = ['NOT ATTACKED', 'UNSUCCESS', 'SUCCESS']
        # print(state, results[state], actions[state], file=sys.stderr)
        for i in range(len(results[state])):
            # print(results[state][i], file=sys.stderr)
            sum_p += results[state][i][0]
            if sum_p < p:
                continue
            next_state = results[state][i][1]
            attack_status = attack_by_mm[results[state][i][3]]
            chosen_action = actions[state]
            break
        
        assert sum_p >= p
        # f.write('ACTION: ', chosen_action)
        s += 'ACTION: {}    '.format(chosen_action)
        s += 'MM_ATTACKED: {}   '.format(attack_status)
        s += '\n'
        f.write(s)
        state = next_state
        it += 1
    f.close()

def convert_keys(obj):
    new_obj = {}
    for key in obj.keys():
        new_obj[str(list(key))] = obj[key]
    
    return new_obj

def save_data(V, state_best_actions, state_best_actions_responses, all_action_responses):
    final_dict = {
        "V": V.tolist(),
        "best_action": convert_keys(state_best_actions),
        "best_action_result": convert_keys(state_best_actions_responses),
        "action_results": convert_keys(all_action_responses)
    }
    path = "VI_iteration_data.json"
    json_object = json.dumps(final_dict, indent=4)
    with open(path, 'w+') as file:
        file.write(json_object)


def value_iteration(start = None, sim = False):
    V = np.zeros((MAX_POSITIONS, MAX_MATERIALS + 1, MAX_ARROWS + 1, MAX_MM_STATES, MAX_MM_HEALTH + 1))
    state_best_actions = {}
    state_best_actions_responses = {}
    state_all_actions_responses = {}
    iteration = 0
    over = False
    while not over:
        new_V = np.zeros(V.shape)
        max_delta = np.NINF
        print(f"iteration={iteration}")
        for state, _ in np.ndenumerate(V):
            # checking the terminal state, MM health = 0.
            if state[-1] not in MM_HEALTHS:
                continue
            
            if state[-1] == 0:
                print(
                    f"({POS_STRING[state[0]]},{state[1]},{state[2]},{MM_STATES_STRING[state[3]]},{state[4]}):NONE=" + "[{:0.3f}]".format(new_V[state]))
                continue
            
            all_action_results = action(state, V)

            if all_action_results is None:
                assert False

            best_action, best_action_responses, value, all_action_responses = mm_action(state, V, all_action_results)

            state_best_actions[state] = best_action
            state_best_actions_responses[state] = best_action_responses
            state_all_actions_responses[state] = all_action_responses

            new_V[state] = value
            delta = abs(new_V[state] - V[state])
            if delta > max_delta:
                max_delta = delta

            print(
                f"({POS_STRING[state[0]]},{state[1]},{state[2]},{MM_STATES_STRING[state[3]]},{state[4]}):{best_action}="  + "[{:0.3f}]".format(new_V[state]))

        print(max_delta, ',', file=sys.stderr)
        V = deepcopy(new_V)
        if max_delta < BELLMON_ERROR:
            if sim:
                save_data(V, state_best_actions, state_best_actions_responses, state_all_actions_responses)
                Simulate(V, state_best_actions, state_best_actions_responses, start)
            over = True
        iteration += 1

# pos, mat, arrow, mm_state, health
value_iteration((POS_C, 2, 0, MM_READY, 100), True)
# value_iteration((POS_W,0,0,MM_DORMANT,100), True)
