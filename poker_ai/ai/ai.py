import copy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Union, Optional

import joblib
import numpy as np

from poker_ai.ai.agent import Agent, PersistentAgent
from poker_ai.games.short_deck.state import ShortDeckPokerState


log = logging.getLogger("sync.ai")


# TODO most of these methods part of agent?
def update_strategy(
    agent: Agent,
    state: ShortDeckPokerState,
    agent_player: int,
    iteration: int,
):
    """
    Update pre-flop strategy using a more theoretically sound approach.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    agent_player : int
        The Player.
    iteration : int
        The iteration.
    locks : TrainingLocks
        The locks for multiprocessing
    """
    player_not_in_hand = not state.players[agent_player].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0:
        return

    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However, we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, player_i, iteration)

    playing_player = state.player_i
    if playing_player == agent_player:
        # Choose action
        action = agent.act(state.info_set, state.legal_actions)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")

        # Increment the action counter.
        with agent.locks.actions_count:
            actions_count = agent.actions_count.get(
                state.info_set, {action: 0 for action in state.legal_actions}
            )
            actions_count[action] += 1
            agent.actions_count[state.info_set] = actions_count

        new_state = state.apply_action(action)
        update_strategy(agent, new_state, agent_player, iteration)
        return

    # Otherwise traverse every action
    for action in state.legal_actions:
        log.debug(f"Going to Traverse {action} for opponent")
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, agent_player, iteration)


def cfr(
    agent: Agent,
    state: ShortDeckPokerState,
    agent_player: int,
    iteration: int,
    pruning_threshold: Optional[int] = None,
) -> float:
    """
    Regular counter-factual regret minimization algorithm.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    agent_player : int
        The Player.
    iteration : int
        The iteration.
    pruning_threshold : Optional[int]
        The threshold for pruning the tree. If none then no pruning is done.
    """
    log.debug("CFR")
    log.debug("########")
    log.debug(f"Iteration: {iteration}")
    log.debug(f"Player Set to Update Regret: {agent_player}")
    log.debug(f"P(h): {state.player_i}")
    log.debug(f"P(h) Updating Regret? {state.player_i == agent_player}")
    log.debug(f"Betting Round {state._betting_stage}")
    log.debug(f"Community Cards {state._table.community_cards}")
    for player_i, player in enumerate(state.players):
        log.debug(f"Player {player_i} hole cards: {player.cards}")
    try:
        log.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    log.debug(f"Betting Action Correct?: {state.players}")

    player_not_in_hand = not state.players[agent_player].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[agent_player]

    # NOTE(fedden): The logic in Algorithm 1 in the supplementary material
    #               instructs the following lines of logic, but state class
    #               will already skip to the next in-hand player.
    # elif p_i not in hand:
    #   cfr()
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfr()
    default_regret = {a: 0.0 for a in state.legal_actions}
    strategy = agent.calc_curr_strategy_for_info_set(
        state.info_set, state.legal_actions
    )
    log.debug(f"Calculated Strategy for {state.info_set}: {strategy}")

    playing_player = state.player_i
    if playing_player == agent_player:
        # TODO: Does updating strategy here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        info_set_val = 0.0
        action_vals: Dict[str, float] = {}
        regret = agent.regret.get(state.info_set, default_regret)
        for action in state.legal_actions:
            if pruning_threshold is not None and regret[action] < pruning_threshold:
                # Prune the path
                continue

            log.debug(
                f"ACTION TRAVERSED FOR REGRET: player {agent_player} ACTION: {action}"
            )
            new_state = state.apply_action(action)
            action_vals[action] = cfr(
                agent, new_state, agent_player, iteration, pruning_threshold
            )
            log.debug(f"Got EV for {action}: {action_vals[action]}")
            info_set_val += strategy[action] * action_vals[action]
            log.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {state.info_set}\n"
                f"STRATEGY: {strategy[action]}: {strategy[action] * action_vals[action]}"
            )

        log.debug(f"Updated EV at {state.info_set}: {info_set_val}")
        with agent.locks.regret:
            regret = agent.regret.get(state.info_set, default_regret)
            for action, action_val in action_vals.items():
                regret[action] += action_val - info_set_val

            agent.regret[state.info_set] = regret

        return info_set_val

    action = np.random.choice(list(strategy.keys()), p=list(strategy.values()))
    log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
    new_state = state.apply_action(action)
    return cfr(agent, new_state, agent_player, iteration)


def serialise(
    agent: Agent,
    save_path: Path,
    iteration: int,
    server_state: Dict[str, Union[str, float, int, None]],
):
    """
    Write progress of optimising agent (and server state) to file.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    save_path : ShortDeckPokerState
        Current game state.
    iteration : int
        The iteration.
    server_state : Dict[str, Union[str, float, int, None]]
        All the variables required to resume training.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    # Load the shared strategy that we accumulate into.
    agent_path = os.path.abspath(str(save_path / f"agent.joblib"))
    persistent_agent = PersistentAgent()
    if os.path.isfile(agent_path):
        try:
            with agent.locks.file:
                persistent_agent = joblib.load(agent_path)
        except Exception:
            log.warning(f"Failed to load agent from file {agent_path}")
            return

    # Lock shared dicts so no other process modifies it whilst writing to
    # file.
    # Calculate the strategy for each info sets regret, and accumulate in
    # the offline agent's strategy.
    with agent.locks.regret:
        strategies = agent.calc_all_current_strategies()

    for info_set, strategy in strategies.items():
        if info_set not in persistent_agent.post_flop_strategy:
            persistent_agent.post_flop_strategy[info_set] = {
                a: 0 for a in strategy.keys()
            }

        for action, prob in strategy.items():
            # += prob because we're interested in average of strategy snapshots
            persistent_agent.post_flop_strategy[info_set][action] += prob

    with agent.locks.regret:
        persistent_agent.regret = copy.deepcopy(agent.regret)

    # TODO what are other places where this lock is used?
    with agent.locks.actions_count:
        persistent_agent.pre_flop_strategy = copy.deepcopy(agent.actions_count)

    with agent.locks.file:
        joblib.dump(persistent_agent, agent_path)

    # Dump the server state to file too, but first update a few bits of the
    # state so when we load it next time, we start from the right place in
    # the optimisation process.
    server_path = save_path / f"server.gz"
    server_state["agent_path"] = agent_path
    server_state["start_timestep"] = iteration + 1
    joblib.dump(server_state, server_path)
