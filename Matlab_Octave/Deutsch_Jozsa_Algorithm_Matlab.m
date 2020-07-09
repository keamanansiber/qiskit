% Deutsch-Jozsa Algorithm
% Hatma Suryotrisongko
% http://keamanansiber.id

qubit_0 = [1; 0]
qubit_1 = [0; 1]
Identity_gate = eye(2)
Hadamard_gate = [ 1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2) ]

balance_A_function = [0 1 0 0 0 0 0 0; 1 0 0 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1]
balance_B_function = [1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0]
constant_D_function = [0 1 0 0 0 0 0 0; 1 0 0 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0]

% Change this as you like!
Nx = 2
SecretFunction_gate = balance_A_function


input_x = [1]
for k=1:Nx
  input_x = kron(input_x, qubit_0);
end
input_xy = kron(input_x, qubit_1);

Hx = [1]
for k=1:Nx
  Hx = kron(Hx, Hadamard_gate);
end

Hxy = kron(Hx, Hadamard_gate);

DeutschJozsaAlgorithm = kron(Hx,Identity_gate) * SecretFunction_gate * Hxy


result = DeutschJozsaAlgorithm * input_xy
