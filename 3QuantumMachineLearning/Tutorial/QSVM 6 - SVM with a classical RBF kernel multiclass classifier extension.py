import numpy as np
from qiskit.ml.datasets import wine
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua.algorithms import SklearnSVM
from qiskit.aqua.components.multiclass_extensions import (OneAgainstRest,
                                                          AllPairs,
                                                          ErrorCorrectingCode)


feature_dim = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = wine(training_size=20,
                                                              test_size=10, n=feature_dim, plot_data=True)

temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)


extensions = [ OneAgainstRest(),
               AllPairs(),
               ErrorCorrectingCode(code_size=5) ]

for extension in extensions:
    result = SklearnSVM(training_input, test_input, total_array, multiclass_extension=extension).run()
    print("\n----- Using multiclass extension: '{}' -----\n".format(extension.__class__.__name__))
    for k,v in result.items():
        print("'{}' : {}".format(k, v))



# 'testing_accuracy' : 1.0

# sumber = https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/machine_learning/svm_classical_multiclass.ipynb
