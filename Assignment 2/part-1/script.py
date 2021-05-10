from pprint import pprint

MAX_STEPS = 10
INF = 1e100

transition_table = {
    "A": {
        "R": {"B": [0.8, -1], "A": [0.2, -1]},
        "U": {
            "C": [0.8, -1],
            "A": [0.2, -1],
        },
    },
    "B": {
        "L": {"A": [0.8, -1], "B": [0.2, -1]},
        "U": {
            "R": [0.8, -4],
            "B": [0.2, -1],
        },
    },
    "C": {
        "R": {"R": [0.25, -3], "C": [0.75, -1]},
        "D": {
            "A": [0.8, -1],
            "C": [0.2, -1],
        },
    },
}

final_reward = 19
# final_reward = 16.5
gamma = 0.20
delta = 0.01
V = [{"A": 0, "B": 0, "C": 0, "R": final_reward}]


def termination(steps):
    if steps > MAX_STEPS:
        return True
    elif steps == 0:
        return False
    else:
        diff = [
            abs(V[steps]["A"] - V[steps - 1]["A"]),
            abs(V[steps]["B"] - V[steps - 1]["B"]),
            abs(V[steps]["C"] - V[steps - 1]["C"]),
        ]
        print(steps, diff)
        has_converged = True
        for val in diff:
            if val >= delta:    
                has_converged = False
                break
        print('----------------------------------------')
        print("in loop", has_converged)
        return has_converged


def calc_V(cur_V):
    next_V = {"A": 0, "B": 0, "C": 0, "R": final_reward}
    """state -> s"""
    for state in transition_table:
        max_val = -INF
        state_def = transition_table[state]
        for action in state_def:
            action_def = state_def[action]
            total_val = 0
            print("{} : {} ->".format(state, action), end="")
            for next_state in action_def:
                next_state_def = action_def[next_state]
                val = next_state_def[0] * (
                    next_state_def[1] + gamma * cur_V[next_state]
                )
                print(
                    "{}*({} + {}*{})".format(
                        next_state_def[0],
                        next_state_def[1],
                        gamma,
                        cur_V[next_state],
                    ),
                    end=" ",
                )
                total_val += val
            print("=", total_val)
            max_val = max(max_val, total_val)
        next_V[state] = max_val
        print("max:", max_val)

    return next_V


cur_steps = 0
while not termination(cur_steps):
    cur_V = V[-1]
    next_V = calc_V(cur_V)
    V.append(next_V)
    cur_steps += 1


for ind, iterations in enumerate(V):
    print(ind)
    pprint(iterations)