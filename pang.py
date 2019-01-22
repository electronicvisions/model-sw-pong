import nest
import cProfile
import argparse
import pong
import numpy as np
import pickle
import time
import os
import logging
import gzip
from copy import copy

#: float: Amplitude of STDP curve in arbitrary units.
STDP_AMPLITUDE = 36.0
#: float: Time constant of STDP curve in milliseconds.
STDP_TAU = 64.
#: int: Cutoff for accumulated STDP.
STDP_SATURATION = 128
#: bool: Consider only causal part of STDP curve.
ONLY_CAUSAL = True
#: int: Scaling factor that affects simulation time and length of spiketrain.
EXPERIMENT_LENGTH_SCALING = 1
#: int: Amount of time the network is simulated in milliseconds.
POLL_TIME = 200 * EXPERIMENT_LENGTH_SCALING
#: int: Number of spikes for input spiketrain.
NO_SPIKES = 20 * EXPERIMENT_LENGTH_SCALING
#: float: Inter-Spike Interval (ISI) of input spiketrain.
ISI = 10.
#: bool: Indicates whether to inject current noise into each neuron.
BACKGROUND = True
#: float: Standard deviation of Gaussian current noise in picoampere.
BG_STD = 100.
#: string: Indicates whether to use a uniform ("uniform") or Poissonian ("possion") input spiketrain.
TRAIN_TYPE = "uniform"
#: float: Initial weight when using uniform initial weight distribution.
WEIGHT = 10.0
#: float: Saturation weight.
WEIGHT_MAX = 63.0
#: float: Scaling factor for weight transformation.
WEIGHT_SCALE = 400.0 / 63
#: float: Offset for weight transformation.
WEIGHT_OFFSET = 32.
#: string: Indicates whether to use uniform or gaussian initial weight distribution.
WEIGHT_DIST = "gaussian"
#: float: Average initial weight for Gaussian distribution.
WEIGHT_MEAN = 14.
#: float: Standard deviation of Gaussian initial weight distribution.
WEIGHT_STD = 2.
#: float: Learning rate to use in weight updates.
LEARNING_RATE = 0.125
#: float: Used as factor when updating expected reward.
MEAN_RUNS = 2.0
#: dict: Dictionary with neuron parameters.
NEURON_DICT = {
    "tau_m": 28.53,
    "V_th": 1278.,
    "E_L": 616.,
    "V_m": 616.,
    "V_reset": 355.,
    "t_ref": 3.98,
    "tau_syn_ex": 1.8,
    "C_m": 2.36,
}
#: float: Reward offset.
REWARD_OFFSET = 0.


class Network(object):
    """Represents the spiking neural network.

        Args:
            num_neurons (int): Number of neurons to use.
            num_threads (int): Number of threads to use.
            initial_weights (list, numpy.ndarray): Initial weight matrix.
            with_voltmeters (bool): Create and attach voltmeters.
    """

    def __init__(self,
                 num_neurons=32,
                 num_threads=1,
                 initial_weights=None,
                 with_voltmeters=False):
        self.num_threads = num_threads
        self.num_neurons = num_neurons
        self.initial_weights = initial_weights
        self.with_voltmeters = with_voltmeters
        self.neuron_params = NEURON_DICT
        self.reset_network(initial=True)
        self.weights = [self.get_weights(x) for x in range(self.num_neurons)]

    def get_weights(self, neuron):
        """Get weights targeting a specific neuron.

        Args:
            neuron (int): Number of targeted neuron.

        Returns:
            numpy.ndarray of weights targeting given neuron in DLS weight domain.
        """
        conns = nest.GetConnections(self.input_neurons,
                                    target=[self.motor_neurons[neuron]])
        conn_vals = nest.GetStatus(conns, ["weight"])
        conn_vals = np.array(conn_vals)
        return self.nest2dls(conn_vals)

    def get_all_weights_flat(self):
        """Get all weights as flattened array.

        Returns:
            numpy.array of all weights in DLS weight domain.
        """
        conns = nest.GetConnections(self.input_neurons)
        weights = nest.GetStatus(conns, "weight")
        return self.nest2dls(weights)

    def set_all_weights_flat(self, weights):
        """Set all weights using given list.

        Args:
            weights (list, numpy.array): Flat list or array to set weights to.
        """
        conns = nest.GetConnections(self.input_neurons)
        nest.SetStatus(conns, [{"weight": self.dls2nest(w)} for w in weights])

    def set_weights(self, weights, neuron):
        """Set weights of specific neuron.

        Args:
            weights (list, numpy.array): Weights to set.
            neuron (int): Number of neuron.
        """
        conns = nest.GetConnections(self.input_neurons,
                                    target=[self.motor_neurons[neuron]])
        for conn, wgt in zip(conns, weights):
            nest.SetStatus([conn], {"weight": float(self.dls2nest(wgt))})

    def get_rates(self):
        """Get rates from spike detectors.

        Returns:
            numpy.array of neuronal spike rates.
        """
        events = np.array(nest.GetStatus(self.spikedetector, ["n_events"]))
        events = [x[0] for x in events]
        return np.array(events)

    def get_voltage_traces(self):
        """Get voltage traces from voltmeters.

        Returns:
            List of voltage traces for all neurons.
        """
        if not self.with_voltmeters:
            raise RuntimeError("Network was initialized without voltmeters.")
        traces = []
        for vmeter in self.voltmeter:
            traces.append(nest.GetStatus([vmeter], "events")[0]["V_m"])
        return traces

    def get_activation_function(self, start, stop, step=1):
        """Get activation function for range of weights.

        Args:
            start (int): Starting weight.
            stop (int): Stopping weight.
            step (int): Step size for weights.

        Returns:
            List of list containing spike rates at different weights.
        """
        logging.debug("Getting activation function.")
        weights = np.arange(start, stop, step=step)
        act_fct = []
        for weight in weights:
            logging.debug("Getting rates at weight %f..." % weight)
            logging.debug("NEST weight: %f" % self.dls2nest(weight))
            self.reset_network(initial=True)
            self.set_all_weights_flat(
                np.diag([weight for _ in range(self.num_neurons)]).flatten())
            spikes = [1 + x * ISI for x in range(NO_SPIKES)]
            nest.SetStatus(self.input_generator, {'spike_times': spikes})
            self.run_simulation()
            rates = self.get_rates()
            logging.debug("Got rates:")
            logging.debug(rates)
            act_fct.append(rates)
        return act_fct

    def get_spiketrains(self):
        """Extract spike times from spikedetector.

        Returns:
            List of list containing spike times from all neurons.
        """

        events = np.array(nest.GetStatus(self.spikedetector, ["events"]))
        out = [[] for nrn in range(self.num_neurons)]
        for neuron, neuron_events in enumerate(events):
            for sp_time in neuron_events[0]['times']:
                out[neuron].append(sp_time)
        return out

    def dls2nest(self, dls_weight):
        """Convert weights from DLS into NEST domain using linear transformation.

        Args:
            dls_weight (list, int, numpy.array): Weight to convert.

        Returns:
            Weight in NEST domain.
        """
        dls_weight = np.array(dls_weight)
        dls_weight += (dls_weight > 0) * WEIGHT_OFFSET
        dls_weight *= WEIGHT_SCALE
        return dls_weight

    def nest2dls(self, nest_weight):
        """Convert weight from NEST into DLS domain using linear transformation.

        Args:
            nest_weight(list, int, numpy.array): Weight to convert.

        Returns:
            Weight in DLS domain.
        """
        nest_weight = np.array(nest_weight)
        nest_weight /= WEIGHT_SCALE
        nest_weight -= (nest_weight > 0) * WEIGHT_OFFSET
        return nest_weight

    def calculate_stdp(self, pre_spikes, post_spikes,
                       only_causal=True,
                       next_neighbor=True):
        """Calculates STDP trace for given spike trains.

        Args:
            pre_spikes(list, numpy.array): Presynaptic spike times in milliseconds.
            post_spikes(list, numpy.array): Postsynaptic spike times in milliseconds.
            only_causal (bool): Use only causal part.
            next_neighbor (bool): Use only next-neighbor coincidences.

        Returns:
            Scalar that corresponds to accumulated STDP trace.
        """
        pre_spikes, post_spikes = np.sort(pre_spikes), np.sort(post_spikes)
        facilitation = 0
        depression = 0
        positions = np.searchsorted(pre_spikes, post_spikes)
        last_position = -1
        for spike, position in zip(post_spikes, positions):
            if position == last_position and next_neighbor:
                continue  # only next-neighbor pairs
            if position > 0:
                before_spike = pre_spikes[position - 1]
                facilitation += STDP_AMPLITUDE * np.exp(-(spike - before_spike)
                                                        / STDP_TAU)
            if position < len(pre_spikes):
                after_spike = pre_spikes[position]
                depression += STDP_AMPLITUDE * np.exp(-(after_spike - spike) /
                                                      STDP_TAU)
            last_position = position
        if only_causal:
            return min(facilitation, STDP_SATURATION)
        else:
            return min(facilitation - depression, STDP_SATURATION)

    def create_input_spiketrain(self):
        """Create input spiketrain.

        Returns:
            numpy.array of spike times.
        """
        if TRAIN_TYPE == "uniform":
            spikes = [1 + x * ISI for x in range(NO_SPIKES)]
        elif TRAIN_TYPE == "poisson":
            spikes = [np.random.poisson(ISI)]
            while spikes[-1] < POLL_TIME:
                spikes.append(spikes[-1] + np.random.poisson(ISI))
            spikes = np.array(spikes).astype(float)
        return spikes

    def set_input_spiketrain(self, input_cell):
        """Set spike train encoding position of ball along y-axis.

        Args:
            input_cell (int): Input unit that corresponds to ball position.
        """
        self.spikes = self.create_input_spiketrain()
        # Reset first
        for input_neuron in range(self.num_neurons):
            nest.SetStatus([self.input_generator[input_neuron]],
                           {'spike_times': []})
        nest.SetStatus([self.input_generator[input_cell]],
                       {'spike_times': self.spikes})

    def run_simulation(self):
        """Run NEST simulation."""
        self.weights = []
        for neuron in range(self.num_neurons):
            self.weights.append(self.get_weights(neuron))
        nest.Simulate(POLL_TIME)

    def apply_reward(self, reward, ball_neuron):
        """Apply given reward by calculating and applying weight updates.

        Args:
            reward (float): Reward.
            ball_neuron (int): Input neuron that corresponds to the ball's cell (only this unit has transmitted spiketrain).
        """
        for connection in nest.GetConnections(
            [self.input_neurons[ball_neuron]]):
            # iterate connections originating from input neuron
            # connection[0]: source, connection[1]: target

            input_neuron = connection[0]
            motor_neuron = connection[1]
            input_gen = nest.GetConnections(self.input_generator,
                                            target=[input_neuron])[0][0]
            pre_spikes = self.spikes
            post_detector = nest.GetConnections([motor_neuron],
                                                target=self.spikedetector)[0][
                                                    1
                                                ]
            post_events = nest.GetStatus([post_detector], "events")
            post_spikes = []
            for sp_time in post_events[0]["times"]:
                post_spikes.append(sp_time)
            correlation = self.calculate_stdp(pre_spikes, post_spikes,
                                              only_causal=ONLY_CAUSAL)
            old_weight = self.nest2dls(np.array(nest.GetStatus([connection],
                                                               "weight"))[0])
            new_weight = np.round(
                old_weight + LEARNING_RATE * correlation * reward)
            if new_weight > 63.:
                new_weight = 63.
            nest.SetStatus([connection], {"weight": self.dls2nest(new_weight)})

    def reset_rng(self):
        """Reset RNG using new seed."""
        nest.SetStatus([0], {
            'rng_seeds': np.random.randint(2 ** 32 - 1,
                                           size=self.num_threads).tolist()
        })

    def reset_network(self, initial=False):
        """Reset network and NEST objects.

        Args:
            initial (bool): If false, weights will be conserved.
        """
        if not initial:
            weights = self.get_all_weights_flat()
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": self.num_threads})
        self.reset_rng()
        self.input_neurons = nest.Create("parrot_neuron", self.num_neurons)
        self.input_generator = nest.Create("spike_generator", self.num_neurons)
        self.motor_neurons = nest.Create("iaf_psc_exp", self.num_neurons,
                                         params=self.neuron_params)
        self.spikedetector = nest.Create("spike_detector", self.num_neurons)
        if BACKGROUND:
            self.background_generator = nest.Create("noise_generator",
                                                    self.num_neurons,
                                                    params={"std": BG_STD})
        nest.Connect(self.motor_neurons, self.spikedetector,
                     {'rule': 'one_to_one'})
        if WEIGHT_DIST == "uniform":
            nest.Connect(self.input_neurons, self.motor_neurons,
                         {"rule": 'all_to_all'},
                         {"weight": self.dls2nest(WEIGHT)})
        elif WEIGHT_DIST == "gaussian":
            nest.Connect(
                self.input_neurons, self.motor_neurons,
                {"rule": 'all_to_all'}, {
                    "weight": self.dls2nest(np.round(np.random.normal(
                        WEIGHT_MEAN, WEIGHT_STD,
                        size=(self.num_neurons, self.num_neurons))))
                })
        nest.Connect(self.input_generator, self.input_neurons,
                     {'rule': 'one_to_one'})
        if self.with_voltmeters:
            self.voltmeter = nest.Create("voltmeter", self.num_neurons)
            nest.SetStatus(self.voltmeter, {"withgid": True, "withtime": True})
            nest.Connect(self.voltmeter, self.motor_neurons,
                         {'rule': 'one_to_one'})
        if BACKGROUND:
            nest.Connect(self.background_generator, self.motor_neurons,
                         {'rule': 'one_to_one'})
        nest.set_verbosity("M_WARNING")
        if not initial:
            self.set_all_weights_flat(weights)
        elif self.initial_weights is not None:
            self.set_all_weights_flat(np.array(self.initial_weights).flatten())


class AIPong:
    """Combines neural network and pong game.

        Args:
            num_games (int): Maximum number of games to play.
            debug (bool): Print debug messages.
            num_threads (int): Number of threads to use.
            initial_weights (numpy.array): Initial weight matrix.
    """

    def __init__(self,
                 num_games=100000,
                 debug=False,
                 num_threads=1,
                 initial_weights=None):
        self.game = pong.GameOfPong(debug=debug)
        self.initial_weights = initial_weights
        self.network = Network(num_threads=num_threads,
                               initial_weights=initial_weights)
        self.debug = debug
        self.num_games = num_games
        self.num_threads = num_threads

    def poll_network(self):
        """Get grid cell network wants to move to. Find this cell by finding the winning (highest rate) motor neuron.
        """
        logging.debug("Running simulation...")
        self.network.run_simulation()
        rates = self.network.get_rates()
        logging.debug("Got rates: ")
        logging.debug(rates)
        winning_neuron = int(
            np.random.choice(np.flatnonzero(rates == rates.max())))
        self.target_cell = winning_neuron

    def adjust_paddle_movement(self):
        """Adjust paddle movement according to target cell.
        """
        if self.game.right_paddle.get_cell()[1] < self.target_cell:
            self.game.move_right_paddle_up()
        if self.game.right_paddle.get_cell()[1] == self.target_cell:
            self.game.dont_move_right_paddle()
        if self.game.right_paddle.get_cell()[1] > self.target_cell:
            self.game.move_right_paddle_down()

    def reward_network_by_move(self):
        """ Reward network based on whether the correct cell was targeted.
        """
        index = self.ball_cell

        def calc_reward(bare_reward):
            self.reward = bare_reward + REWARD_OFFSET
            if self.mean_reward[index] == -1:
                self.mean_reward[index] = self.reward
            self.success = self.reward - self.mean_reward[index]
            self.mean_reward[index] = float(
                self.mean_reward[index] + self.success / MEAN_RUNS)
            self.performance[index] = np.ceil(bare_reward)
            self.network.apply_reward(self.success, index)

        rewards_dict = {0: 1., 1: 0.7, 2: 0.4, 3: 0.1}
        distance = np.abs(self.target_cell - self.ball_cell)
        if distance in rewards_dict:
            calc_reward(rewards_dict[distance])
        else:
            calc_reward(0)
        logging.debug("Applying reward=%.3f, mean reward=%.3f, success=%.3f" %
                      (self.reward, self.mean_reward[index], self.success))
        logging.debug("Mean rewards:")
        logging.debug(self.mean_reward)
        logging.debug("Average mean reward:")
        logging.debug(np.mean(self.mean_reward))
        logging.debug("Performances:")
        logging.debug(self.performance)

    def get_parameters(self):
        """Get used parameters.

        Returns:
            dict: Dictionary of used parameters.
        """
        parameter_dict = {
            "WEIGHT_SCALE": WEIGHT_SCALE,
            "WEIGHT": WEIGHT,
            "ISI": ISI,
            "BG_STD": BG_STD,
            "NO_SPIKES": NO_SPIKES,
            "WEIGHT_MAX": WEIGHT_MAX,
            "STDP_AMPLITUDE": STDP_AMPLITUDE,
            "STDP_TAU": STDP_TAU,
            "WEIGHT_DIST": WEIGHT_DIST,
            "WEIGHT_MEAN": WEIGHT_MEAN,
            "WEIGHT_STD": WEIGHT_STD,
            "NEURON_DICT": NEURON_DICT,
            "MEAN_RUNS": MEAN_RUNS,
            "ONLY_CAUSAL": ONLY_CAUSAL,
            "REWARD_OFFSET": REWARD_OFFSET,
        }
        return parameter_dict

    def run_games(self, save_every=0, folder="", max_runs=np.inf):
        """Run games by polling network and stepping through game.

        Args:
            save_every (int): Number of iterations after which to save variables.
            folder (string): Folder to save to.
            max_runs (int): Maximum number of iterations.
        """
        self.run = 0
        expdir = os.path.join(folder, str(time.time()))
        parameters = self.get_parameters()
        if save_every != 0:
            if folder != "":
                if not os.path.exists(folder):
                    os.mkdir(folder)
            os.mkdir(expdir)
            file = open(os.path.join(expdir, "parameters.pkl"), "w")
            pickle.dump(parameters, file)
            file.close()
        self.correlations = []
        self.weight_history = []
        self.mean_reward = np.array([-1.
                                     for _ in range(self.network.num_neurons)])
        self.performance = np.array([-1.
                                     for _ in range(self.network.num_neurons)])
        self.mean_reward_history = []
        self.performance_history = []
        self.run_history = []
        while self.run < max_runs:
            self.ball_cell = self.game.ball.get_cell()[1]
            logging.debug(
                "Run #%d, Ball in cell (%d, %d), Paddle in cell (%d, %d)" %
                (self.run, self.game.ball.get_cell()[0],
                 self.game.ball.get_cell()[1],
                 self.game.right_paddle.get_cell()[0],
                 self.game.right_paddle.get_cell()[1]))
            self.network.set_input_spiketrain(self.ball_cell)
            self.poll_network()
            logging.debug("Network wants to go to cell %d" % self.target_cell)
            self.adjust_paddle_movement()
            self.game.step()
            self.reward_network_by_move()
            self.weight_history.append(copy(self.network.weights))
            self.mean_reward_history.append(copy(self.mean_reward))
            self.performance_history.append(copy(self.performance))
            self.network.reset_network()
            self.run_history.append(self.run)
            self.run += 1
            if self.game.result != pong.NO_WIN:
                logging.debug("Game ended with %d" % (self.game.result))
                self.game = pong.GameOfPong(debug=self.debug)
        if save_every:
            with gzip.open(os.path.join(expdir, "data.pkl.gz"), "w") as file:
                pickle.dump((self.mean_reward_history[::save_every],
                             self.performance_history[::save_every],
                             self.weight_history[::save_every],
                             self.run_history[::save_every]), file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",
                        type=int,
                        default=20000,
                        help="Number of runs to perform.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Verbose debugging output.")
    parser.add_argument("--save-every",
                        type=int,
                        default=0,
                        help="Save every X runs to file.")
    parser.add_argument("--act-fct",
                        nargs=3,
                        type=float,
                        help="Get activation function: start stop step.")
    parser.add_argument("--threads",
                        type=int,
                        help="Number of threads for NEST simulation.",
                        default=1)
    parser.add_argument("--folder",
                        type=str,
                        default="",
                        help="Folder to save experiments to.")
    parser.add_argument("--diagonal-weights",
                        action="store_true",
                        help="Use diagonal initial weights.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.diagonal_weights:
        initial_weights = np.diagflat(np.ones(32)) * 63.
    else:
        initial_weights = None
    print initial_weights
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.act_fct:
        network = Network(num_threads=args.threads)
        act_fct = network.get_activation_function(args.act_fct[0],
                                                  args.act_fct[1],
                                                  step=args.act_fct[2])
        if args.save_every:
            f = open("rates_%f.pkl" % time.time(), "w")
            pickle.dump(act_fct, f)
            f.close()
    else:
        aipong = AIPong(debug=True,
                        num_threads=args.threads,
                        initial_weights=initial_weights)
        if args.profile:
            cProfile.run(
                "aipong.run_games(max_runs=args.runs, save_every=args.save_every)",
                sort="time")
        else:
            aipong.run_games(max_runs=args.runs,
                             save_every=args.save_every,
                             folder=args.folder)
