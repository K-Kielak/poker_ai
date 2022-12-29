import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Union, List

import enlighten

from poker_ai.ai.agent import Agent
from poker_ai import utils
from poker_ai.config import MAX_WORKERS
from poker_ai.games.short_deck import state
from poker_ai.ai.multiprocess.worker import Worker

log = logging.getLogger("sync.server")
manager = mp.Manager()


@dataclass
class WorkerStatus:
    status: str
    iterationOfLastUpdate: int
    timeOfLastUpdate: int


class Server:
    """Server class to manage all workers optimising CFR algorithm."""

    def __init__(
        self,
        strategy_interval: int,
        n_iterations: int,
        lcfr_threshold: int,
        discount_interval: int,
        prune_threshold: int,
        c: int,
        n_players: int,
        dump_iteration: int,
        update_threshold: int,
        print_status_interval: int,
        save_path: Union[str, Path],
        lut_path: Union[str, Path] = ".",
        agent_path: Optional[Union[str, Path]] = None,
        sync_update_strategy: bool = False,
        sync_cfr: bool = False,
        sync_discount: bool = False,
        sync_serialise: bool = False,
        start_timestep: int = 1,
        n_processes: int = MAX_WORKERS,
    ):
        """Set up the optimisation server."""
        self._strategy_interval = strategy_interval
        self._n_iterations = n_iterations
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._prune_threshold = prune_threshold
        self._c = c
        self._n_players = n_players
        self._dump_iteration = dump_iteration
        self._update_threshold = update_threshold
        self._print_status_interval = print_status_interval
        self._save_path = save_path
        self._lut_path = lut_path
        self._agent_path = agent_path
        self._sync_update_strategy = sync_update_strategy
        self._sync_cfr = sync_cfr
        self._sync_discount = sync_discount
        self._sync_serialise = sync_serialise
        self._start_timestep = start_timestep
        self._info_set_lut: state.InfoSetLookupTable = utils.io.load_info_set_lut(
            lut_path
        )
        log.info("Loaded lookup table.")

        self._job_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=n_processes)
        self._status_queue: mp.Queue = mp.Queue()
        self._logging_queue: mp.Queue = mp.Queue()

        self._worker_status: Dict[str, WorkerStatus] = dict()
        self._job_times: Dict[str, List[int]] = dict()

        self._agent: Agent = Agent(agent_path)
        self._locks: Dict[str, mp.synchronize.Lock] = dict(
            regret=mp.Lock(),
            strategy=mp.Lock(),
            pre_flop_strategy=mp.Lock(),
            file=mp.Lock(),
        )
        if os.environ.get("TESTING_SUITE"):
            n_processes = 4
        self._workers: Dict[str, Worker] = self._start_workers(n_processes)

    def search(self):
        """Perform MCCFR and train the agent.

        If all `sync_*` parameters are set to True then there shouldn't be any
        difference between this and the original MCCFR implementation.
        """
        log.info(f"synchronising update_strategy - {self._sync_update_strategy}")
        log.info(f"synchronising cfr             - {self._sync_cfr}")
        log.info(f"synchronising discount        - {self._sync_discount}")
        log.info(f"synchronising serialise_agent - {self._sync_serialise}")
        log.info(f"Parent PID: {os.getpid()}")
        progress_bar_manager = enlighten.get_manager()
        progress_bar = progress_bar_manager.counter(
            total=self._n_iterations, desc="Optimisation iterations", unit="iter"
        )
        for iteration in range(self._start_timestep, self._n_iterations + 1):
            # Log any messages from the worker in this master process to avoid
            # weirdness with tqdm.
            self._update_jobs_and_workers_status(iteration)
            while not self._logging_queue.empty():
                log.info(self._logging_queue.get())

            # Optimise for each player's position.
            for player_i in range(self._n_players):
                if (
                    iteration > self._update_threshold
                    and iteration % self._strategy_interval == 0
                ):
                    self.job(
                        "update_strategy",
                        sync_workers=self._sync_update_strategy,
                        iteration=iteration,
                        player_i=player_i,
                    )
                self.job(
                    "cfr",
                    sync_workers=self._sync_cfr,
                    iteration=iteration,
                    player_i=player_i,
                )
            if (
                iteration < self._lcfr_threshold
                and iteration % self._discount_interval == 0
            ):
                self.job(
                    "discount", sync_workers=self._sync_discount, iteration=iteration
                )
            if (
                iteration > self._update_threshold
                and iteration % self._dump_iteration == 0
            ):
                self.job(
                    "serialise",
                    sync_workers=self._sync_serialise,
                    iteration=iteration,
                    server_state=self.to_dict(),
                )
            if iteration % self._print_status_interval == 0:
                self._print_status(iteration)
            progress_bar.update()

    def terminate(self, safe: bool = True):
        """Kill all workers."""
        # Wait for all workers to finish their current jobs.
        self._job_queue.join()
        # Ensure all workers are idle.
        self._wait_until_all_workers_are_idle()
        # Send the terminate command to all workers.
        for _ in self._workers.values():
            name = "terminate"
            kwargs = dict()
            self._job_queue.put((name, kwargs), block=True)
            log.info("sending sentinel to worker")
        for name, worker in self._workers.items():
            worker.join()
            log.info(f"worker {name} joined.")

    def to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        """Serialise the server object to save the progress of optimisation."""
        config = dict(
            strategy_interval=self._strategy_interval,
            n_iterations=self._n_iterations,
            lcfr_threshold=self._lcfr_threshold,
            discount_interval=self._discount_interval,
            prune_threshold=self._prune_threshold,
            c=self._c,
            n_players=self._n_players,
            dump_iteration=self._dump_iteration,
            update_threshold=self._update_threshold,
            save_path=self._save_path,
            lut_path=self._lut_path,
            agent_path=self._agent_path,
            sync_update_strategy=self._sync_update_strategy,
            sync_cfr=self._sync_cfr,
            sync_discount=self._sync_discount,
            sync_serialise=self._sync_serialise,
            start_timestep=self._start_timestep,
        )
        # Sort dictionary for human-friendlyness and convert all pathlib.Path
        # objects to absolute path strings.
        return {
            k: os.path.abspath(str(v)) if isinstance(v, Path) else v
            for k, v in sorted(config.items())
        }

    @staticmethod
    def from_dict(config):
        """Load serialised server and return instance."""
        return Server(**config)

    def job(self, job_name: str, sync_workers: bool = False, **kwargs):
        """
        Create a job for the workers.

        ...

        Parameters
        ----------
        job_name : str
            Name of job.
        sync_workers : bool
            Whether or not to synchronize workers.
        """
        func = self._syncronised_job if sync_workers else self._send_job
        func(job_name, **kwargs)

    def _send_job(self, job_name: str, **kwargs):
        """Send job of type `name` with arguments `kwargs` to worker pool."""
        self._job_queue.put((job_name, kwargs), block=True)

    def _syncronised_job(self, job_name: str, **kwargs):
        """Only perform this job with one process."""
        # Wait for all enqueued jobs to be completed.
        self._job_queue.join()
        # Wait for all workers to become idle.
        self._wait_until_all_workers_are_idle()
        log.info(f"Sending synchronised {job_name} to workers")
        log.info(self._worker_status)
        # Send the job to a single worker.
        self._send_job(job_name, **kwargs)
        # Wait for the synchronised job to be completed.
        self._job_queue.join()
        # The status update of the worker starting the job should be flushed
        # first.
        name_a, status = self._status_queue.get(block=True)
        assert status == job_name, f"expected {job_name} but got {status}"
        # Next get the status update of the job being completed.
        name_b, status = self._status_queue.get(block=True)
        assert status == "idle", f"status should be idle but was {status}"
        assert name_a == name_b, f"{name_a} != {name_b}"

    def _start_workers(self, n_processes: int) -> Dict[str, Worker]:
        """Begin the processes."""
        workers = dict()
        for _ in range(n_processes):
            worker = Worker(
                job_queue=self._job_queue,
                status_queue=self._status_queue,
                logging_queue=self._logging_queue,
                locks=self._locks,
                agent=self._agent,
                info_set_lut=self._info_set_lut,
                n_players=self._n_players,
                prune_threshold=self._prune_threshold,
                c=self._c,
                lcfr_threshold=self._lcfr_threshold,
                discount_interval=self._discount_interval,
                update_threshold=self._update_threshold,
                dump_iteration=self._dump_iteration,
                save_path=self._save_path,
            )
            workers[worker.name] = worker
        for name, worker in workers.items():
            worker.start()
            log.info(f"started worker {name}")
        return workers

    def _wait_until_all_workers_are_idle(self, sleep_secs=0.5):
        """Blocks until all workers have finished their current job."""
        while True:
            # Read all status updates.
            while not self._status_queue.empty():
                worker_name, status = self._status_queue.get(block=False)
                # Iteration -1 to indicate that status has been
                # read outside of iteration loop
                self._worker_status[worker_name] = WorkerStatus(status, -1, time.time())

            # Are all workers idle, all workers statues obtained, if so, stop waiting.
            all_idle = all(
                status.status == "idle" for status in self._worker_status.values()
            )
            all_statuses_obtained = len(self._worker_status) == len(self._workers)
            if all_idle and all_statuses_obtained:
                break
            time.sleep(sleep_secs)
            log.info(
                {w: s for w, s in self._worker_status.items() if s.status != "idle"}
            )

    def _print_status(self, iteration: int):
        log.info("Jobs statistics:")
        jobs_str = ""
        for name, times in self._job_times.items():
            jobs_str += (
                f"\t- {name.upper()}: "
                f"done {len(times)} times, "
                f"max time {np.max(times):.2f}s, "
                f"last 10 avg time {np.mean(times[-10:]):.2f}s, "
                f"avg time {np.mean(times):.2f}s\n"
            )
        log.info(jobs_str)

        log.info("Current worker status:")
        workers_str = ""
        for name, status in self._worker_status.items():
            last_update = self._worker_status[name]
            passed_iterations = iteration - last_update.iterationOfLastUpdate
            passed_time = timedelta(seconds=time.time() - last_update.timeOfLastUpdate)
            workers_str += (
                f"\t- {name}: {status.status}; "
                f"Last change was {passed_iterations} iterations "
                f"and {passed_time} (hh:mm:ss) ago.\n"
            )
        log.info(workers_str)

    def _update_jobs_and_workers_status(self, iteration: int):
        while not self._status_queue.empty():
            worker_name, status = self._status_queue.get(block=False)

            if worker_name in self._worker_status:
                last_status = self._worker_status[worker_name]
                if last_status.status not in self._job_times:
                    self._job_times[last_status.status] = []

                passed_time = time.time() - last_status.timeOfLastUpdate
                self._job_times[last_status.status].append(passed_time)

            # Update worker status
            self._worker_status[worker_name] = WorkerStatus(
                status, iteration, time.time()
            )
