from typing import List, Tuple


class Action:
    pass


class ContinuousAction(Action):
    v : float
    w : float
    beep: int

    def __init__(self, v, w, beep=0):
        self.v = v
        self.w = w

    def reverse(self):
        return [self.v, self.w]


class DiscreteActions:
    actions: List[ContinuousAction] = []

    def __init__(self, actions: List[Tuple]):
        for action in actions:
            assert action[0] >= 0
            assert len(action) == 2
            self.actions.append(ContinuousAction(*action))

    def __len__(self):
        return len(self.actions)
 
    def __getitem__(self, index):
        return self.actions[index]
