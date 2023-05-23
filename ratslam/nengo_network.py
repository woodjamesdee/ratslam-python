import numpy as np
from ratslam._globals import *
import nengo

def inhibit(t):
    return 2.0 if t >= 0.0 else 0.0

class NengoPoseCells:
    """
    """

    def __init__(self, update_func, active_func, ensemble_size=64, seed=12345678, filename=None, perform_inhibition=False):
        """
        """

        self.model = nengo.Network()
        with self.model:

            # Nodes
            self.update_input = nengo.Node(update_func) # gets view_cell data, vtrans and vrot (input to the pose_cell network)
            self.active_input = nengo.Node(active_func) # this gets the current active pose_cell (used for training)
            self.inhibit_input = nengo.Node(inhibit) # used for inhibition (aka stopping training)

            # Ensenbles
            self.pre_ensemble = nengo.Ensemble(ensemble_size, dimensions=6, seed=seed) # update_input connects to this ensemble
            self.post_ensemble = nengo.Ensemble(ensemble_size, dimensions=3) # pre_ensemble connects to this ensemble (learning occurs over connection)
            self.error_ensemble = nengo.Ensemble(ensemble_size, dimensions=3) # this ensemble provides the error signal for training

            # Connections
            self.input_pre_connection = nengo.Connection(self.update_input, self.pre_ensemble)
            #weight_matrix = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]
            weight_matrix = [[0]*ensemble_size, [0]*ensemble_size, [0]*ensemble_size]
            if filename is None:
                self.pre_post_connection = nengo.Connection(self.pre_ensemble.neurons, self.post_ensemble, transform=weight_matrix)
            else:
                try:
                    weights = np.load(filename)
                    self.pre_post_connection = nengo.Connection(self.pre_ensemble.neurons, self.post_ensemble, transform=weights)
                    print(f"Loaded weights from {filename}")
                except:
                    self.pre_post_connection = nengo.Connection(self.pre_ensemble.neurons, self.post_ensemble, transform=weight_matrix)
                    print(f"Failed to load weights from {filename}, using default construction")
            self.pre_post_connection.learning_rule_type = nengo.PES()
            self.error_signal_connection = nengo.Connection(self.error_ensemble, self.pre_post_connection.learning_rule)
            self.cells_input_connection = nengo.Connection(self.active_input, self.error_ensemble, transform=-1)
            self.post_error_connection = nengo.Connection(self.post_ensemble, self.error_ensemble)

            if perform_inhibition:
                nengo.Connection(self.inhibit_input, self.error_ensemble.neurons, transform=[[-1]] * self.error_ensemble.n_neurons)

            # Probes
            self.active_probe = nengo.Probe(self.active_input) # to view the belief of the normal pose cell network
            self.pre_probe = nengo.Probe(self.pre_ensemble, synapse=0.01) # to view the state of the pre_ensemble
            self.post_probe = nengo.Probe(self.post_ensemble, synapse=0.01) # to view the state of the post_ensemble
            self.error_probe = nengo.Probe(self.error_ensemble, synapse=0.03) # to view the state of the error ensemble
            self.weights_probe = nengo.Probe(self.pre_post_connection, "weights") # to get the connection weights (decoders)

            # Simulator
            self.simulator = nengo.Simulator(self.model, progress_bar=False)

    def run(self, time):
        self.simulator.run(time)

    def save(self, filename):
        try:
            np.save(filename, self.simulator.data[self.weights_probe][-1])
            print(f"Saved weights to {filename}")
        except Exception as e:
            print(f"Failed to save weights to {filename}")
            print(e)

