#!/usr/bin/env python
import argparse
from gettext import install
import itertools as itt
from copy import copy

import one_to_one


def ifmt(i):
    return '_'.join('pos' if x.value else 'neg' for x in i)


def sfmt(s):
    return f'agent_r{s.agent}_human_r{s.human}_{ifmt(s.intentions)}'


def afmt(a):
    return a.value


def ofmt(o):
    return o.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shopping')
    parser.add_argument('n', help="The number of rooms", type=int, default=2)
    # parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--symbolic', action='store_true')
    config = parser.parse_args()

    # TODO change size to width and height
    assert config.n > 1
    assert 0 < config.gamma <= 1

    pos_space = one_to_one.RangeSpace(config.n)
    int_space = one_to_one.JointNamedSpace(
        # pos/neg
        **{f'r{n}': one_to_one.RangeSpace(2) for n in range(config.n)}
    )
    state_space = one_to_one.JointNamedSpace(
        agent=pos_space, human=pos_space, intentions=int_space)

    actions = ['report'] + [f'goto_r{n}' for n in range(config.n)] + [
        f'inform_r{n}' for n in range(config.n)] + [f'ask_r{n}' for n in range(config.n)]
    action_space = one_to_one.DomainSpace(actions)

    obs = 'nan', 'neg', 'pos'
    obs_space = one_to_one.DomainSpace(obs)

    # print('states')
    # for s in state_space.elems:
    #     print(sfmt(s))

    # print('actions')
    # for a in action_space.elems:
    #     print(afmt(a))

    # print('observations')
    # for o in obs_space.elems:
    #     print(ofmt(o))

    # import sys
    # sys.exit(0)
    assert state_space.nelems == config.n ** 2 * 2 ** config.n
    assert action_space.nelems == 1 + config.n + config.n + config.n
    assert obs_space.nelems == 3

    print(
        """# Showroom Environment without noise;
#
# The agent navigates a house with n rooms being tasked with the goal of convincing a human
# to have interests in all rooms. the human follows the agent and will be convinced only
# when they are at the same room. Observations regarding the human's interests in rooms can be
# obtained and should ideally be memorized during navigation.
#
# State-space (n * n * 2**n) : position of the agent in the house (n rooms),
# position of the human in the house (n rooms), 
# and proposition of the human interests in each room (2 ** n).
#
# Action-space (1 + 3*n) : movements {goto_r(n)}, query info on a human interest {`ask_r(n)`},
# inform info to the human {`inform_r(n)`} and report to terminate a trial in current
# room {`report`}.
#
# Observation-space (3) : No information or no feedback from the human `nan`,
# a negative feedback `neg`, and a positive feedback `pos`."""
    )

    print(f'#')
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print()
    print('values: reward')

    print()
    print(f'# states: {state_space.nelems}')
    print(f'states: {" ".join(sfmt(s) for s in state_space.elems)}') if config.symbolic else print(
        f'states: {state_space.nelems}')

    print()
    # print(f'actions: {" ".join(action_space.values)}')
    print(f'# actions: {action_space.nelems}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}') if config.symbolic else print(
        f'actions: {action_space.nelems}')

    print()
    print(f'# observations: {obs_space.nelems}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}') if config.symbolic else print(
        f'observations: {obs_space.nelems}')

    start_states = [
        s
        for s in state_space.elems
    ]
    # pstart_states = 1 / len(start_states)

    # START
    print()
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}') if config.symbolic else print(
        f'start include: {" ".join(str(s.idx) for s in start_states)}')
    # print(f'start include: uniform')

    # TRANSITIONS
    print()
    for idx_a, a in enumerate(action_space.elems):
        if 'ask' in a.value:
            print(f'T: \t{afmt(a)} \tidentity') if config.symbolic \
                else print(f'T: \t{idx_a} \tidentity')
        else:
            for idx_s, s in enumerate(state_space.elems):
                if a.value == 'report':
                    print(f'T: \t{afmt(a)}: \t{sfmt(s)} \treset') if config.symbolic \
                        else print(f'T: \t{idx_a}: \t{idx_s} \treset')
                elif 'goto' in a.value:
                    s1 = copy(s)
                    for n in range(config.n):
                        if str(n) in a.value:
                            s1.agent.value = n
                            if s.agent.value == s.human.value:
                                s1.human.value = n
                            print(
                                f'T: \t{afmt(a)}: \t{sfmt(s)}: \t{sfmt(s1)} \t1.0')
                elif 'inform' in a.value:
                    s1 = copy(s)
                    for n, s1_intention in enumerate(s1.intentions):
                        if str(n) in a.value:
                            if s.agent.value == s.human.value == n:
                                s1_intention.value = 1
                            print(
                                f'T: \t{afmt(a)}: \t{sfmt(s)}:\t {sfmt(s1)} \t1.0')

    # OBSERVATIONS
    print()
    for a, s1, o in itt.product(action_space.elems, state_space.elems, obs_space.elems):
        if 'ask' not in a.value and o.value == 'nan':
            print(f'O: \t{afmt(a)}: \t{sfmt(s1)}: \t{ofmt(o)} \t1.0')
        if 'ask' in a.value:
            if (
                str(s1.agent.value) in a.value
                and s1.agent.value == s1.human.value
                and s1.intentions.value.__getattribute__(a.value[4:]) == 1
                and o.value == 'pos'
            ):
                print(f'O: \t{afmt(a)}: \t{sfmt(s1)}: \t{ofmt(o)} \t1.0')
            elif (
                str(s1.agent.value) in a.value
                and s1.agent.value == s1.human.value
                and s1.intentions.value.__getattribute__(a.value[4:]) == 0
                and o.value == 'neg'
            ):
                print(f'O: \t{afmt(a)}: \t{sfmt(s1)}: \t{ofmt(o)} \t1.0')
            elif (
                str(s1.agent.value) not in a.value
                and s1.agent.value == s1.human.value
                and o.value == 'nan'
            ):
                print(f'O: \t{afmt(a)}: \t{sfmt(s1)}: \t{ofmt(o)} \t1.0')
            elif (
                s1.agent.value != s1.human.value
                and o.value == 'nan'
            ):
                print(f'O: \t{afmt(a)}: \t{sfmt(s1)}: \t{ofmt(o)} \t1.0')

    # REWARDS
    print()
    for a, s in itt.product(action_space.elems, state_space.elems):
        if a.value == 'report':
            if all(s.intentions.value):
                print(f'R: \t{afmt(a)}: \t{sfmt(s)}: \t*: \t* \t1.0')
            else:
                print(f'R: \t{afmt(a)}: \t{sfmt(s)}: \t*: \t* \t-1.0')
        else:
            print(f'R: \t{afmt(a)}: \t{sfmt(s)}: \t*: \t* \t-0.01')
