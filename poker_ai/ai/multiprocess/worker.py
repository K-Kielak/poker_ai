import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from poker_ai.ai import ai
from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.games.short_deck import state


class Worker(mp.Process):
    """Subclass of multiprocessing Process to handle agent optimisation."""

    def __init__(
        self,
        job_queue: mp.Queue,
        status_queue: mp.Queue,
        logging_queue: mp.Queue,
        agent: Agent,
        info_set_lut: state.InfoSetLookupTable,
        n_players: int,
        prune_threshold: int,
        c: int,
        lcfr_threshold: int,
        discount_interval: int,
        update_threshold: int,
        dump_iteration: int,
        save_path: Path,
    ):
        """Construct the process, setup the state."""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._status_queue: mp.Queue = status_queue
        self._logging_queue: mp.Queue = logging_queue
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._save_path = Path(save_path)
        self._info_set_lut: state.InfoSetLookupTable = info_set_lut
        self._setup_new_game()

    def run(self):
        """Compute the next job the server sent."""
        # Seed process.
        self._set_seed()
        # Start processing loop, that will block on a wait for the next job
        # that will be sent from the server to be consumed by the worker(s).
        loop_i = 0
        while True:
            # Get the name of the method and the key word arguments needed for
            # the method.
            self._update_status("idle")
            name, kwargs = self._job_queue.get(block=True)
            if name == "terminate":
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "discount":
                function = self._discount
            elif name == "update_strategy":
                function = self._update_strategy
            elif name == "serialise":
                function = self._serialise
            else:
                raise ValueError(f"Unrecognised function name: {name}")

            self._update_status(name)
            function(**kwargs)
            # Notify the job queue that the task is done.
            self._job_queue.task_done()
            loop_i += 1

    def _set_seed(self):
        """Lose all reproducability as we need unique streams per worker."""
        # NOTE(fedden): NumPy in particular has a problem with processes and
        #               seeds: https://github.com/numpy/numpy/issues/9650
        random_seed = int.from_bytes(os.urandom(4), byteorder="little")
        utils.random.seed(random_seed)

    def _cfr(self, iteration, player_i):
        """Search over random game and calculate the strategy."""
        self._setup_new_game()
        use_pruning = np.random.uniform() < 0.95
        pruning_allowed = iteration > self._prune_threshold
        if pruning_allowed and use_pruning:
            ai.cfr(
                agent=self._agent,
                state=self._state,
                agent_player=player_i,
                iteration=iteration,
                pruning_threshold=self._c,
            )
        else:
            ai.cfr(
                agent=self._agent,
                state=self._state,
                agent_player=player_i,
                iteration=iteration,
            )

    def _discount(self, iteration):
        """Discount previous regrets and strategy."""
        # TODO(fedden): Is discount_interval actually set/managed in
        #               minutes here? In Algorithm 1 this should be managed
        #               in minutes using perhaps the time module, but here
        #               it appears to be being managed by the iterations
        #               count.
        discount_factor = (iteration / self._discount_interval) / (
            (iteration / self._discount_interval) + 1
        )
        self._agent.discount(discount_factor)

    def _update_strategy(self, iteration, player_i):
        """Update the strategy."""
        ai.update_strategy(self._agent, self._state, player_i, iteration)

    def _serialise(
        self, iteration: int, server_state: Dict[str, Union[str, float, int, None]]
    ):
        """Write progress of optimising agent (and server state) to file."""
        ai.serialise(
            agent=self._agent,
            save_path=self._save_path,
            iteration=iteration,
            server_state=server_state,
        )

    def _update_status(self, status, log_status: bool = False):
        """Update the status of this worker by posting it to the server."""
        if log_status:
            self._logging_queue.put(
                f"{self.name} updating status to {status}", block=True
            )
        self._status_queue.put((self.name, status), block=True)

    def _setup_new_game(self):
        """Setup up new poker game."""
        self._state: state.ShortDeckPokerState = state.new_game(
            self._n_players,
            self._info_set_lut,
        )
