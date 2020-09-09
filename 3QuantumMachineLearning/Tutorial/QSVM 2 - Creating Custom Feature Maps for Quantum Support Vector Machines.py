import numpy as np
import matplotlib.pyplot as plt
import functools

from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap, PauliFeatureMap
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.feature_maps import self_product
from qiskit.aqua.algorithms import QSVM
from qiskit.ml.datasets import ad_hoc_data


# Generate synthetic training and test sets from the SecondOrderExpansion quantum feature map
feature_dim = 2
sample_Total, training_dataset, test_dataset, class_labels = ad_hoc_data(training_size=20, test_size=10,
                                                                         n=feature_dim, gap=0.3,
                                                                         plot_data=False)

# Using the statevector simulator
backend = BasicAer.get_backend('statevector_simulator')
random_seed = 10598

quantum_instance = QuantumInstance(backend, seed_simulator=random_seed, seed_transpiler=random_seed)


# Generate the feature map
feature_map = ZFeatureMap(feature_dimension=feature_dim, reps=2)

# Run the Quantum Kernel Estimator and classify the test data
qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset, test_dataset=test_dataset)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])


feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)

qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset, test_dataset=test_dataset)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])


def custom_data_map_func(x):
    """Define a function map from R^n to R.

    Args:
        x (np.ndarray): data
    Returns:
        double: the mapped value
    """
    coeff = x[0] if len(x) == 1 else functools.reduce(lambda m, n: m * n, np.pi - x)
    return coeff


feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, data_map_func=custom_data_map_func)

qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset, test_dataset=test_dataset)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])


feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=2, paulis = ['Z','Y','ZZ'])

qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset, test_dataset=test_dataset)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])


feature_dim = 3
sample_Total_b, training_dataset_b, test_dataset_b, class_labels = ad_hoc_data(training_size=20, test_size=10,
                                                                               n=feature_dim, gap=0.3,
                                                                               plot_data=False)


feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=2,
                              paulis = ['Y','Z','ZZ','ZZZ'], data_map_func=custom_data_map_func)

qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset_b, test_dataset=test_dataset_b)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])

from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import BlueprintCircuit


class CustomFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""

    def __init__(self, feature_dimension, depth=2, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
        """
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
        self._entangler_map = None
        if self._entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in
                                   range(i + 1, self._feature_dimension)]

    def construct_circuit(self, x, qr, inverse=False):
        """Construct the feature map circuit.

        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit.
            inverse (bool): whether or not to invert the circuit.

        Returns:
            QuantumCircuit: a quantum circuit transforming data x.
        """
        qc = QuantumCircuit(qr)

        for _ in range(self._depth):
            for i in range(self._feature_dimension):
                qc.rx(x[i], qr[i])
            for [source, target] in self._entangler_map:
                qc.cx(qr[source], qr[target])
                qc.u1(x[source] * x[target], qr[target])
                qc.cx(qr[source], qr[target])

        if inverse:
            qc.inverse()
        return qc


feature_map = CustomFeatureMap(feature_dimension=2, depth=1)

qsvm = QSVM(feature_map=feature_map, training_dataset=training_dataset, test_dataset=test_dataset)

result = qsvm.run(quantum_instance)
print("testing success ratio: ", result['testing_accuracy'])



# ValueError: cannot reshape array of size 1 into shape (40,1)

# testing success ratio:  0.55

# The <class '__main__.CustomFeatureMap'> object as input for the QSVM is deprecated as of 0.7.0 and will
#                 be removed no earlier than 3 months after the release.

# sumber = https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/machine_learning/custom_feature_map.ipynb
