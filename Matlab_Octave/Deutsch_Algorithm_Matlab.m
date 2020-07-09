% Deutsch’s Algorithm
% Hatma Suryotrisongko
% http://keamanansiber.id

balance_A_function = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
balance_B_function = [ 0 1 0 0; 1 0 0 0; 0 0 1 0; 0 0 0 1 ]
constant_C_function = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
constant_D_function = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]

% choose a function above, for the secret function
% then, simply measure the first qubit.
% If it is in state |0>, then we know that f is a CONSTANT function, 
% otherwise it is a BALANCED function.

U_secret_function = balance_B_function
qubit_0 = [1; 0]
qubit_1 = [0; 1]
Identity_gate = eye(2)
Hadamard_gate = [ 1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2) ]
DeutschAlgorithm = ( kron(Hadamard_gate, Identity_gate) ) *  (U_secret_function *  (kron(Hadamard_gate, Hadamard_gate) *  (kron(qubit_0, qubit_1)) ) )
