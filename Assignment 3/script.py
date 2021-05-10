# storing states as ((a, b), (c,d), e)
from copy import deepcopy
from enum import Enum
from typing import List, Tuple


class State:
    def __init__(self, a_x, a_y, t_x, t_y, call):
        self.a_col: int = a_x
        self.a_row: int = a_y
        self.t_col: int = t_x
        self.t_row: int = t_y
        self.call: int = call

    def get_id(self):
        idx = self.a_row * 64 + self.a_col * 16 + self.t_row * 8 + self.t_col * 2 + self.call
        return idx

    def __str__(self):
        return f"S_{self.a_row}{self.a_col}_{self.t_row}{self.t_col}_{self.call}"

    def get_a_pos(self):
        return self.a_col, self.a_row

    def set_a_pos(self, x):
        self.a_col = x[0]
        self.a_row = x[1]

    def increase_x(self):
        self.a_col = min(self.a_col + 1, 3)

    def decrease_x(self):
        self.a_col = max(self.a_col - 1, 0)

    def increase_y(self):
        self.a_row = min(self.a_row + 1, 1)

    def decrease_y(self):
        self.a_row = max(self.a_row - 1, 0)

    def t_increase_x(self):
        self.t_col = min(self.t_col + 1, 3)

    def t_decrease_x(self):
        self.t_col = max(self.t_col - 1, 0)

    def t_increase_y(self):
        self.t_row = min(self.t_row + 1, 1)

    def t_decrease_y(self):
        self.t_row = max(self.t_row - 1, 0)


class Actions(Enum):
    STAY, UP, DOWN, LEFT, RIGHT = range(5)


class Observations(Enum):
    o1, o2, o3, o4, o5, o6 = range(6)


class POMDP:
    def __init__(self, states):
        self.states = states

    def apply_transition(self, state: State, action):
        results = []
        # Agent Movements
        # print("T1", state, action)
        if action == Actions.RIGHT:
            new_state = deepcopy(state)
            new_state.increase_x()
            results.append((x, new_state))
            new_state = deepcopy(state)
            new_state.decrease_x()
            results.append((1 - x, new_state))
        elif action == Actions.LEFT:
            new_state = deepcopy(state)
            new_state.increase_x()
            results.append((1 - x, new_state))
            new_state = deepcopy(state)
            new_state.decrease_x()
            results.append((x, new_state))
        elif action == Actions.DOWN:
            new_state = deepcopy(state)
            new_state.increase_y()
            results.append((x, new_state))
            new_state = deepcopy(state)
            new_state.decrease_y()
            results.append((1 - x, new_state))
        elif action == Actions.UP:
            new_state = deepcopy(state)
            new_state.increase_y()
            results.append((1 - x, new_state))
            new_state = deepcopy(state)
            new_state.decrease_y()
            results.append((x, new_state))
        elif action == Actions.STAY:
            results.append((1, state))
        movements_results = []
        # APPLY TARGET MOVEMENT
        # print("T2", state, action)

        for pr, st in results:
            next_state = deepcopy(st)
            # STAY
            movements_results.append((0.6 * pr, next_state))
            # RIGHT
            next_state = deepcopy(st)
            next_state.t_increase_x()
            movements_results.append((0.1 * pr, next_state))
            # LEFT
            next_state = deepcopy(st)
            next_state.t_decrease_x()
            movements_results.append((0.1 * pr, next_state))
            # DOWN
            next_state = deepcopy(st)
            next_state.t_increase_y()
            movements_results.append((0.1 * pr, next_state))
            # UP
            next_state = deepcopy(st)
            next_state.t_decrease_y()
            movements_results.append((0.1 * pr, next_state))
        # print("T3", state, action)
        # FILTER MOVEMENT RESULTS
        filter_movements = []
        unique_states = set([st.get_id() for pr, st in movements_results])
        for st in unique_states:
            prt = 0.0
            for pr, st2 in movements_results:
                if st2.get_id() == st:
                    prt += pr
            filter_movements.append((prt, deepcopy(self.states[st])))

        final_results: List[Tuple[float, State]] = []
        # print("T4", state, action)
        if state.call == 0:
            for pr, st in filter_movements:
                # CALL IS ON
                next_state = deepcopy(st)
                final_results.append((pr * 0.5, next_state))
                # CALL IS OFF
                next_state = deepcopy(st)
                next_state.call = 1
                final_results.append((pr * 0.5, next_state))
        elif state.call == 1 and (state.a_row, state.a_col) != (state.t_row, state.t_col):
            # print("     FF", state, action)
            for pr, st in filter_movements:
                # CALL IS ON
                next_state = deepcopy(st)
                final_results.append((pr * 0.9, next_state))
                # CALL IS OFF
                next_state = deepcopy(st)
                next_state.call = 0
                final_results.append((pr * 0.1, next_state))
        elif state.call == 1 and (state.a_row, state.a_col) == (state.t_row, state.t_col):
            print("call is 0,", state, action)
            for pr, st in filter_movements:
                # CALL IS OFF
                next_state = deepcopy(st)
                next_state.call = 0
                final_results.append((pr, next_state))

        total = 0.0
        for (pr, _) in final_results:
            total += pr
        assert 0.99 < total < 1.01
        return final_results

    def transition_table(self):
        for stt in self.states:
            for action in Actions:
                # print("State action pair", stt, action)
                rez = self.apply_transition(stt, action)
                for pr, res in rez:
                    print(f"T: {action.name} : {str(stt)} : {str(res)} {pr}", file=f)

    def observation_table(self):
        results = []
        for end_state in self.states:
            pos_diff = (end_state.a_col - end_state.t_col, end_state.a_row - end_state.t_row)
            if pos_diff == (0, 0):
                results.append((Observations.o1, end_state))
            elif pos_diff == (-1, 0):
                # target is to right
                results.append((Observations.o2, end_state))
            elif pos_diff == (1, 0):
                # target it to left
                results.append((Observations.o4, end_state))
            elif pos_diff == (0, -1):
                # target is to down
                results.append((Observations.o3, end_state))
            elif pos_diff == (0, 1):
                # target is up
                results.append((Observations.o5, end_state))
            else:
                results.append((Observations.o6, end_state))
        for obs, state in results:
            print(f"O : * : {str(state)} : {obs.name} {1.0}", file=f)

    def reward_table(self):
        results = []
        for obs in range(len(Observations)):
            for st in self.states:
                curr_obs = Observations(obs)
                if curr_obs == Observations.o1 and st.call == 1 and (st.a_row, st.a_col) == (st.t_row, st.t_col):
                    results.append((curr_obs, reeward - 1, st))
                else:
                    pass
        print(f"R : * : * : * : * -1", file=f)
        print(f"R : STAY : * : * : * 0", file=f)
        for obs, reward, st in results:
            print(f"R : * : * : {str(st)}: {obs.name} {reeward - 1}", file=f)
        for obs, reward, st in results:
            print(f"R : STAY : * : {str(st)}: {obs.name} {reeward}", file=f)


if __name__ == "__main__":
    discount = 1
    states = []
    roll_no = 2019101050
    x = 1 - ((roll_no % 10000) % 30 + 1) / 100
    count = 0
    reeward = (roll_no % 90 + 10)
    f = open("pulak.pomdp", "w")
    for a_row in range(2):
        for a_col in range(4):
            for t_row in range(2):
                for t_col in range(4):
                    for call in range(2):
                        states.append(State(a_col, a_row, t_col, t_row, call))
                        assert count == states[-1].get_id()
                        count += 1

    num_states = len(states)
    num_actions = len(Actions)
    num_observation = len(Observations)
    print("discount: 0.5", file=f)
    print("values: reward", file=f)
    print("states:", end=" ", file=f)
    for statez in states:
        print(statez, end=" ", file=f)
    print(file=f)
    print("actions:", end=" ", file=f)
    for actionz in Actions:
        print(actionz.name, end=" ", file=f)
    print(file=f)
    print("observations:", end=" ", file=f)
    for obsz in Observations:
        print(obsz.name, end=" ", file=f)
    print(file=f)
    print("start: ", file=f)
    count = 0
    
    # q1
    for state in states:
        pos = (state.t_row, state.t_col)
        a_pos = (state.a_row, state.a_col)
        if pos == (1, 0) and not (a_pos == (0, 0) or a_pos == (1, 1) or a_pos == (1, 0)):
            count += 1
        else:
            pass
    for state in states:
        pos = (state.t_row, state.t_col)
        a_pos = (state.a_row, state.a_col)
        if pos == (1, 0) and not (a_pos == (0, 0) or a_pos == (1, 1) or a_pos == (1, 0)):
            print(state, 1 / count)
            print(1 / count, end=" ", file=f)
        else:
            print(state, 0)
            print(0, end=" ", file=f)
    
    # q2
    # for state in states:
    #     pos = (state.t_row, state.t_col)
    #     a_pos = (state.a_row, state.a_col)
    #     if a_pos == (1, 1) and (pos == (1, 0) or pos == (0, 1) or pos == (1, 2) or pos == (1, 1)) and state.call == 0:
    #         count += 1
    #     else:
    #         pass
    # for state in states:
    #     pos = (state.t_row, state.t_col)
    #     a_pos = (state.a_row, state.a_col)
    #     if a_pos == (1, 1) and (pos == (1, 0) or pos == (0, 1) or pos == (1, 2) or pos == (1, 1)) and state.call == 0:
    #     # if a_pos != (1, 1) or pos != (1, 0) and pos != (0, 1) and pos != (1, 2) and pos != (1, 1) or state.call != 0:
    #         print(state, 1 / count)
    #         print(1 / count, end=" ", file=f)
    #     else:
    #         print(state, 0)
    #         print(0, end=" ", file=f)
    
    # q5
    # for state in states:
    #     pos = (state.t_row, state.t_col)
    #     a_pos = (state.a_row, state.a_col)
    #     if a_pos == (0, 0) and (pos == (0, 1) or pos == (0, 2) or pos == (1, 1) or pos == (1, 2)):
    #         print(state, 0.4 * 0.25 * 0.5)
    #         print(0.4 * 0.25 * 0.5, end=" ", file=f)
    #     elif a_pos == (1, 3) and (pos == (0, 1) or pos == (0, 2) or pos == (1, 1) or pos == (1, 2)):
    #         print(state, 0.6 * 0.25 * 0.5)
    #         print(0.6 * 0.25 * 0.5, end=" ", file=f)
    #     else:
    #         print(0.0, file=f, end=" ")
    print("\n", file=f)
    p = POMDP(states)
    p.transition_table()
    p.observation_table()
    p.reward_table()
    pass
