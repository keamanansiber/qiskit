import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.ml.datasets import wine


import numpy as np
import scipy

from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.aqua.utils import get_feature_dimension

feature_dim = 4 # dimension of each data point
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
aqua_globals.random_seed = random_seed

sample_Total, training_input, test_input, class_labels = wine(
    training_size=training_dataset_size,
    test_size=testing_dataset_size,
    n=feature_dim,
    plot_data=False
)


feature_map = RawFeatureVector(feature_dimension=get_feature_dimension(training_input))
vqc = VQC(COBYLA(maxiter=200),
            feature_map,
            TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
            training_input, test_input)
result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                    seed_simulator=aqua_globals.random_seed,
                                    seed_transpiler=aqua_globals.random_seed))
print("VQC accuracy with RawFeatureVector: ", result['testing_accuracy'])


feature_map = ZZFeatureMap(get_feature_dimension(training_input))
vqc = VQC(COBYLA(maxiter=200),
            feature_map,
            TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
            training_input, test_input)
result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                    seed_simulator=aqua_globals.random_seed,
                                    seed_transpiler=aqua_globals.random_seed))
print("Test accuracy with ZZFeatureMap: ", result['testing_accuracy'])


# VQC accuracy with RawFeatureVector:  0.7
# Test accuracy with ZZFeatureMap:  0.3

# sumber = https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/machine_learning/vqc_feature_map_comparison.ipynb
