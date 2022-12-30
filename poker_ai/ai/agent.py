import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union, Dict, List

import joblib
import numpy as np

log = logging.getLogger("sync.ai")
manager = mp.Manager()


@dataclass
class TrainingLocks:
    regret = mp.Lock()
    actions_count = mp.Lock()
    file = mp.Lock()


@dataclass
class PersistentAgent:
    """Agent information that gets saved on disk"""
    regret: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pre_flop_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    post_flop_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    iteration: int = 0


class Agent:
    """
    Create agent, optionally initialise to agent specified at path.

    ...

    Attributes
    ----------
    actions_count : Dict[str, Dict[str, int]]
        The preflop strategy for an agent.
    regret : Dict[str, Dict[strategy, int]]
        The regret for an agent.
    """

    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".

    def __init__(
        self,
        agent_path: Optional[Union[str, Path]] = None,
        use_manager: bool = True,
    ):
        """Construct an agent."""
        # Don't use manager if we are running tests.
        testing_suite = bool(os.environ.get("TESTING_SUITE", False))
        use_manager = use_manager and not testing_suite
        dict_constructor: Callable = manager.dict if use_manager else dict
        self.actions_count = dict_constructor()
        self.regret = dict_constructor()
        self.locks = TrainingLocks()
        if agent_path is not None:
            saved_agent = joblib.load(agent_path)
            # Assign keys manually because I don't trust the manager proxy.
            for info_set, value in saved_agent["regret"].items():
                self.regret[info_set] = value
            for info_set, value in saved_agent["avg_strategy"].items():
                self.actions_count[info_set] = value

    def act(self, info_set: str, legal_actions: List[str]) -> str:
        """Choose an action given an info set."""
        strategy = self.calc_curr_strategy_for_info_set(info_set, legal_actions)
        log.debug(f"Calculated Strategy for {info_set}: {strategy}")
        return np.random.choice(list(strategy.keys()), p=list(strategy.values()))

    def calc_curr_strategy_for_info_set(
        self, info_set: str, legal_actions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        if info_set not in self.regret:
            if legal_actions is None:
                raise ValueError(
                    "Agent hasn't seen this info set before and no legal "
                    "actions were provided to construct a default one."
                )

            return {action: 1 / len(legal_actions) for action in legal_actions}

        infoset_regret = self.regret[info_set]
        total_regret = sum([max(0, r) for r in infoset_regret.values()])
        if total_regret > 0:
            return {a: max(0, r) / total_regret for a, r in infoset_regret.items()}

        return {a: 1 / len(infoset_regret) for a in infoset_regret.keys()}

    def calc_all_current_strategies(self) -> Dict[str, Dict[str, float]]:
        return {
            info_set: self.calc_curr_strategy_for_info_set(info_set)
            for info_set in self.regret.keys()
        }

    def discount(self, discount_factor: float) -> None:
        with self.locks.regret:
            for info_set in self.regret.keys():
                for action in self.regret[info_set].keys():
                    self.regret[info_set][action] *= discount_factor

        with self.locks.actions_count:
            for info_set in self.actions_count.keys():
                for action in self.actions_count[info_set].keys():
                    self.actions_count[info_set][action] *= discount_factor
