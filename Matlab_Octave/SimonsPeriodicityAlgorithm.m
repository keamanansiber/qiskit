% Simon’s Periodicity Algorithm 
% Hatma Suryotrisongko
% http://keamanansiber.id

m0 = [1; 0]
m1 = [0; 1]

m000000 =  kron( m0,  kron( m0,  kron( m0,  kron( m0,  kron( m0,  m0 )))))
m000001 =  kron( m0,  kron( m0,  kron( m0,  kron( m0,  kron( m0,  m1 )))))
m000010 =  kron( m0,  kron( m0,  kron( m0,  kron( m0,  kron( m1,  m0 )))))
m000011 =  kron( m0,  kron( m0,  kron( m0,  kron( m0,  kron( m1,  m1 )))))
m000100 =  kron( m0,  kron( m0,  kron( m0,  kron( m1,  kron( m0,  m0 )))))
m000101 =  kron( m0,  kron( m0,  kron( m0,  kron( m1,  kron( m0,  m1 )))))
m000110 =  kron( m0,  kron( m0,  kron( m0,  kron( m1,  kron( m1,  m0 )))))
m000111 =  kron( m0,  kron( m0,  kron( m0,  kron( m1,  kron( m1,  m1 )))))
m001000 =  kron( m0,  kron( m0,  kron( m1,  kron( m0,  kron( m0,  m0 )))))
m001001 =  kron( m0,  kron( m0,  kron( m1,  kron( m0,  kron( m0,  m1 )))))
m001010 =  kron( m0,  kron( m0,  kron( m1,  kron( m0,  kron( m1,  m0 )))))
m001011 =  kron( m0,  kron( m0,  kron( m1,  kron( m0,  kron( m1,  m1 )))))
m001100 =  kron( m0,  kron( m0,  kron( m1,  kron( m1,  kron( m0,  m0 )))))
m001101 =  kron( m0,  kron( m0,  kron( m1,  kron( m1,  kron( m0,  m1 )))))
m001110 =  kron( m0,  kron( m0,  kron( m1,  kron( m1,  kron( m1,  m0 )))))
m001111 =  kron( m0,  kron( m0,  kron( m1,  kron( m1,  kron( m1,  m1 )))))
m010000 =  kron( m0,  kron( m1,  kron( m0,  kron( m0,  kron( m0,  m0 )))))
m010001 =  kron( m0,  kron( m1,  kron( m0,  kron( m0,  kron( m0,  m1 )))))
m010010 =  kron( m0,  kron( m1,  kron( m0,  kron( m0,  kron( m1,  m0 )))))
m010011 =  kron( m0,  kron( m1,  kron( m0,  kron( m0,  kron( m1,  m1 )))))
m010100 =  kron( m0,  kron( m1,  kron( m0,  kron( m1,  kron( m0,  m0 )))))
m010101 =  kron( m0,  kron( m1,  kron( m0,  kron( m1,  kron( m0,  m1 )))))
m010110 =  kron( m0,  kron( m1,  kron( m0,  kron( m1,  kron( m1,  m0 )))))
m010111 =  kron( m0,  kron( m1,  kron( m0,  kron( m1,  kron( m1,  m1 )))))
m011000 =  kron( m0,  kron( m1,  kron( m1,  kron( m0,  kron( m0,  m0 )))))
m011001 =  kron( m0,  kron( m1,  kron( m1,  kron( m0,  kron( m0,  m1 )))))
m011010 =  kron( m0,  kron( m1,  kron( m1,  kron( m0,  kron( m1,  m0 )))))
m011011 =  kron( m0,  kron( m1,  kron( m1,  kron( m0,  kron( m1,  m1 )))))
m011100 =  kron( m0,  kron( m1,  kron( m1,  kron( m1,  kron( m0,  m0 )))))
m011101 =  kron( m0,  kron( m1,  kron( m1,  kron( m1,  kron( m0,  m1 )))))
m011110 =  kron( m0,  kron( m1,  kron( m1,  kron( m1,  kron( m1,  m0 )))))
m011111 =  kron( m0,  kron( m1,  kron( m1,  kron( m1,  kron( m1,  m1 )))))
m100000 =  kron( m1,  kron( m0,  kron( m0,  kron( m0,  kron( m0,  m0 )))))
m100001 =  kron( m1,  kron( m0,  kron( m0,  kron( m0,  kron( m0,  m1 )))))
m100010 =  kron( m1,  kron( m0,  kron( m0,  kron( m0,  kron( m1,  m0 )))))
m100011 =  kron( m1,  kron( m0,  kron( m0,  kron( m0,  kron( m1,  m1 )))))
m100100 =  kron( m1,  kron( m0,  kron( m0,  kron( m1,  kron( m0,  m0 )))))
m100101 =  kron( m1,  kron( m0,  kron( m0,  kron( m1,  kron( m0,  m1 )))))
m100110 =  kron( m1,  kron( m0,  kron( m0,  kron( m1,  kron( m1,  m0 )))))
m100111 =  kron( m1,  kron( m0,  kron( m0,  kron( m1,  kron( m1,  m1 )))))
m101000 =  kron( m1,  kron( m0,  kron( m1,  kron( m0,  kron( m0,  m0 )))))
m101001 =  kron( m1,  kron( m0,  kron( m1,  kron( m0,  kron( m0,  m1 )))))
m101010 =  kron( m1,  kron( m0,  kron( m1,  kron( m0,  kron( m1,  m0 )))))
m101011 =  kron( m1,  kron( m0,  kron( m1,  kron( m0,  kron( m1,  m1 )))))
m101100 =  kron( m1,  kron( m0,  kron( m1,  kron( m1,  kron( m0,  m0 )))))
m101101 =  kron( m1,  kron( m0,  kron( m1,  kron( m1,  kron( m0,  m1 )))))
m101110 =  kron( m1,  kron( m0,  kron( m1,  kron( m1,  kron( m1,  m0 )))))
m101111 =  kron( m1,  kron( m0,  kron( m1,  kron( m1,  kron( m1,  m1 )))))
m110000 =  kron( m1,  kron( m1,  kron( m0,  kron( m0,  kron( m0,  m0 )))))
m110001 =  kron( m1,  kron( m1,  kron( m0,  kron( m0,  kron( m0,  m1 )))))
m110010 =  kron( m1,  kron( m1,  kron( m0,  kron( m0,  kron( m1,  m0 )))))
m110011 =  kron( m1,  kron( m1,  kron( m0,  kron( m0,  kron( m1,  m1 )))))
m110100 =  kron( m1,  kron( m1,  kron( m0,  kron( m1,  kron( m0,  m0 )))))
m110101 =  kron( m1,  kron( m1,  kron( m0,  kron( m1,  kron( m0,  m1 )))))
m110110 =  kron( m1,  kron( m1,  kron( m0,  kron( m1,  kron( m1,  m0 )))))
m110111 =  kron( m1,  kron( m1,  kron( m0,  kron( m1,  kron( m1,  m1 )))))
m111000 =  kron( m1,  kron( m1,  kron( m1,  kron( m0,  kron( m0,  m0 )))))
m111001 =  kron( m1,  kron( m1,  kron( m1,  kron( m0,  kron( m0,  m1 )))))
m111010 =  kron( m1,  kron( m1,  kron( m1,  kron( m0,  kron( m1,  m0 )))))
m111011 =  kron( m1,  kron( m1,  kron( m1,  kron( m0,  kron( m1,  m1 )))))
m111100 =  kron( m1,  kron( m1,  kron( m1,  kron( m1,  kron( m0,  m0 )))))
m111101 =  kron( m1,  kron( m1,  kron( m1,  kron( m1,  kron( m0,  m1 )))))
m111110 =  kron( m1,  kron( m1,  kron( m1,  kron( m1,  kron( m1,  m0 )))))
m111111 =  kron( m1,  kron( m1,  kron( m1,  kron( m1,  kron( m1,  m1 )))))

m_input = [  m000000  m000001  m000010  m000011  m000100  m000101  m000110  m000111  m001000  m001001  m001010  m001011  m001100  m001101  m001110  m001111  m010000  m010001  m010010  m010011  m010100  m010101  m010110  m010111  m011000  m011001  m011010  m011011  m011100  m011101  m011110  m011111  m100000  m100001  m100010  m100011  m100100  m100101  m100110  m100111  m101000  m101001  m101010  m101011  m101100  m101101  m101110  m101111  m110000  m110001  m110010  m110011  m110100  m110101  m110110  m110111  m111000  m111001  m111010  m111011  m111100  m111101  m111110  m111111  ]

m_output = [  m000100  m000100  m000100  m000100  m000100  m000100  m000100  m000100  m001001  m001001  m001001  m001001  m001001  m001001  m001001  m001001  m010101  m010101  m010101  m010101  m010101  m010101  m010101  m010101  m011111  m011111  m011111  m011111  m011111  m011111  m011111  m011111  m100001  m100001  m100001  m100001  m100001  m100001  m100001  m100001  m101100  m101100  m101100  m101100  m101100  m101100  m101100  m101100  m110111  m110111  m110111  m110111  m110111  m110111  m110111  m110111  m111101  m111101  m111101  m111101  m111101  m111101  m111101  m111101  ]

U_function_gate  = m_output * inv(m_input) 



% Try it in Matlab/Octave:
qubit_0 = [1; 0]
qubit_1 = [0; 1]
n = 3
input_x = [1]
input_y = [1]
for i=1:n
  input_x = kron(input_x, qubit_0);
  input_y = kron(input_y, qubit_0);
end
input_xy = kron(input_x, input_y)
checkpoint_0 = input_xy

Hadamard_gate = [ 1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2) ]
Identity_gate = eye(2)
Hn = [1]
In = [1]
for i=1:n
  Hn = kron(Hn, Hadamard_gate);
  In = kron(In, Identity_gate);
end
Hn_In = kron(Hn, In);
checkpoint_1 = Hn_In * input_xy

% U_function_gate = ?see next slide?
checkpoint_2 = U_function_gate * checkpoint_1 

checkpoint_3 = Hn_In * checkpoint_2 




