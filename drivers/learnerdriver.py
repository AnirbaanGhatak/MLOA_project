import numpy as np
from collections import defaultdict
from typing import List
from drivers.driver import *


class LearnerDriver(Driver):
    def __init__(self, name, random_action_probability=0, random_action_decay=1, min_random_action_probability=0,
                 speed_rounding=10, discount_factor=0.9, learning_rate=1, max_distance=100):
        super().__init__(name)

        self.q_table = {}
        self.default_q_values = defaultdict(lambda: 0)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.turn_action = 'turn'
        self.correct_headings = defaultdict(lambda: Heading.get_all_headings())

        self.random_action_probability = random_action_probability
        self.min_random_action_probability = min_random_action_probability
        self.random_action_decay = random_action_decay
        self.speed_rounding = speed_rounding
        self.max_distance = max_distance

    def prepare_for_race(self):
        pass            

    def make_a_move(self, car_state: CarState, track_state: TrackState):
        state = self.get_state(car_state, track_state)

        available_actions = self._get_available_actions_for_state(track_state, car_state)

        self._ensure_state_action_in_q_table(state, available_actions)

        if driver_rng().rand() > self.random_action_probability or track_state.distance_ahead < 0:
            value_dict = self.q_table[state]
            max_value = max([value_dict[action] for action in available_actions])
            available_actions = [action for action in available_actions if value_dict[action] == max_value]
            p = np.ones(len(available_actions)) / len(available_actions)

        else:
            v = np.maximum(1 + np.array([self.q_table[state][action] for action in available_actions]), 0)
            if 0 == np.sum(v):
                v = np.ones_like(v)
            p = v / np.sum(v)

        if np.any(np.isnan(p)):
            print(f'NaN detected in p: {p}. value dict = {self.q_table[state]}')
        action = driver_rng().choice(available_actions)

        self.random_action_probability = max(self.random_action_probability * self.random_action_decay,
                                             self.min_random_action_probability)

        if self.turn_action == action:
            return self._choose_turn_direction(track_state)

        else:
            return action

    def _get_available_actions_for_state(self, track_state, car_state):
        if track_state.distance_ahead > 0:
            available_actions = Action.get_sl_actions() 
        else:  
            available_actions = [Action.LightBrake, Action.HeavyBrake, self.turn_action]
        if 0 == car_state.speed:
            available_actions = [action for action in available_actions
                                 if action not in [Action.LightBrake, Action.HeavyBrake, Action.Continue]]
        return available_actions

    def _ensure_state_action_in_q_table(self, state, available_actions: List[Action]):
        if state not in self.q_table:
            self.q_table[state] = {a: self.default_q_values[a] for a in available_actions}  # default q values
        for action in available_actions:
            if action not in self.q_table[state]:
                self.q_table[state][action] = self.default_q_values[action]

    @staticmethod
    def _choose_turn_direction(track_state: TrackState):
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available
            return driver_rng().choice([Action.TurnLeft, Action.TurnRight])  # so choose randomly

        elif track_state.distance_left > 0:  # only left turn
            return Action.TurnLeft

        else:
            return Action.TurnRight  # only right or dead-end

    def _choose_randomly(self, available_actions):
        return driver_rng().choice(available_actions)  # randomly choose an action uniformly over all available actions

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult):

        if result.crashed or result.spun:
            reward = -10000
        elif action in [Action.TurnLeft, Action.TurnRight]:
            reward = max(new_car_state.speed, 1)       # if stationary at end of straight turning is better than braking
        else:
            reward = new_car_state.speed

        new_state = self.get_state(new_car_state, new_track_state)
        if new_state in self.q_table and not previous_track_state.distance_ahead == 0:
            new_state_value_dict = self.q_table[new_state]
            new_state_max_value = max(v for v in new_state_value_dict.values())  # max value over all actions
        else:
            new_state_max_value = 0

        previous_state = self.get_state(previous_car_state, previous_track_state)
        action = self.turn_action if action in [Action.TurnLeft, Action.TurnRight] else action
        if previous_state not in self.q_table:      
            available_actions = self._get_available_actions_for_state(previous_track_state, previous_car_state)
            self.q_table[previous_state] = {a: self.default_q_values[a] for a in available_actions}   # default q values
        if action not in self.q_table[previous_state]:
            print(f'Action {action.name} not in Q table for state {previous_state}')
            self.q_table[previous_state][action] = self.default_q_values[action]

        self.q_table[previous_state][action] += self.learning_rate * (reward +
                                                                      self.discount_factor * new_state_max_value -
                                                                      self.q_table[previous_state][action])

    def get_state(self, car_state: CarState, track_state: TrackState):
        speed = ((car_state.speed - 1) // self.speed_rounding + 1) * self.speed_rounding  
        distance = min(track_state.distance_ahead, self.max_distance)
        return speed, distance

    def update_after_race(self, *args, **kwargs):
        pass

