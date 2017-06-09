#!/usr/bin/env python2

from openpathsampling.pathsimulator import PathSimulator
import openpathsampling as paths
from random import random


class CommittorBootstrap(PathSimulator):
    calc_name = "CommittorBootstrap"

    def __init__(self,
                 trajectory,
                 states,
                 engine,
                 storage,
                 randomizer,
                 initial_guess=None,
                 n_per_snapshot=10):
        super(CommittorBootstrap, self).__init__(storage)

        tmp_network = paths.TPSNetwork.from_states_all_to_all(states)
        subtrajectories = []
        for ens in tmp_network.analysis_ensembles:
                subtrajectories += ens.split(trajectory)
        if len(subtrajectories) == 0:
            raise RuntimeError("No vallid initial trajectory found")

        self.trajectory = subtrajectories[0]
        self.states = states
        self.engine = engine
        self.storage = storage
        self.randomizer = randomizer
        self.n_per_snapshot = n_per_snapshot

        self.initial_state = None
        self.final_state = None
        for state in states:
            if state(self.trajectory[0]):
                self.initial_state = state
            if state(self.trajectory[-1]):
                self.final_state = state
        if initial_guess is None:
            self.snap_frame = len(trajectory)/2
        else:
            self.snap_frame = initial_guess
        self.snap_min = 0
        self.snap_max = len(self.trajectory)-1
        self.random_choice = None
        self.results = {i: {j: [] for j in states if j != i}
                        for i in states}

        self.hash_function = paths.SnapshotByCoordinateDict().hash_function
    def next_frame(self, pBs, snap_min=None, snap_frame=None, snap_max=None):

        if snap_min is None:
            snap_min = self.snap_min
        if snap_frame is None:
            snap_frame = self.snap_frame
        if snap_max is None:
            snap_max = self.snap_max
        for pB in pBs:
            if 0 < pB < 1:
                self.snap_frame = None
                return
            else:
                pass

        for i, state in enumerate(self.states):
            if state is self.initial_state:
                initial_pB = pBs[i]
            elif state is self.final_state:
                final_pB = pBs[i]

        if initial_pB < final_pB:
            snap_max = snap_frame
        elif initial_pB > final_pB:
            snap_min = snap_frame
        else:
            if self.random_choice is None:
                self.random_choice = random()-0.5
            if self.random_choice >= 0:
                snap_min = snap_frame
            elif self.random_choice < 0:
                snap_max = snap_frame

        snap_frame = snap_min+((snap_max-snap_min)/2)
        if snap_max <= snap_frame <= snap_min:
            error_msg = ("No available shooting point left, you might want to" +
                         " increase 'n_per_snapshot'")
            raise RuntimeError(error_msg)
        else:
            self.snap_frame = snap_frame
            self.snap_min = snap_min
            self.snap_max = snap_max

    def committor_values(self, snapshot):
        results = paths.ShootingPointAnalysis(steps=self.storage.steps,
                                              states=self.states)
        committor = [results.committor(state) for state in self.states]
        shash = results.hash_representatives[results.hash_function(snapshot)]
        return [i[shash] for i in committor]

    def make_return_trajs(self, snapshot):
        tmp_dict = {state: [] for state in self.states}

        for traj in self.storage.trajectories:
            if len(traj) == 1:
                continue
            if self.hash_function(traj[0]) == self.hash_function(snapshot):
                for state in self.states:
                    if state(traj[-1]):
                        tmp_dict[state].append(traj)
            elif self.hash_function(traj[-1]) == self.hash_function(snapshot):
                for state in self.states:
                    if state(traj[0]):
                        tmp_dict[state].append(traj.reversed)
            else:
                pass

        for state1 in self.states:
            if len(tmp_dict[state1]) == 0:
                continue
            for state2 in self.states:
                if state2 == state1 or len(tmp_dict[state2]) == 0:
                    continue
                for traj1 in tmp_dict[state1]:
                    for traj2 in tmp_dict[state2]:
                        result_traj = traj1.reversed+traj2[1:]
                        self.results[state1][state2].append(result_traj)
        return self.results

    def run(self):
        while self.snap_frame is not None:
            snap = self.trajectory[self.snap_frame]
            simulation = paths.CommittorSimulation(storage=self.storage,
                                                   engine=self.engine,
                                                   states=self.states,
                                                   randomizer=self.randomizer,
                                                   direction=None,
                                                   initial_snapshots=[snap]
                                                   )
            simulation.run(n_per_snapshot=self.n_per_snapshot)
            pBs = self.committor_values(snap)
            self.next_frame(pBs=pBs)
        return_dict = self.make_return_trajs(snap)
        return return_dict
