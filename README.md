# python
comparing gradient descent with sigmoid and tanh activation functions. 
This code does xor function of two bits and the neural network needs to learn the xor logic.

# source file details
derivative.py plots the derivates of a quadratic equation, a sigmoid and tanh functions.
sigmoidtanhgradientdescent.py shows xor logic without bias for neural network with two inputs 
sigmoidtanhgradientdescentwithbias.py shows xor logic with bias for neural network with xor logic.
xorwiththreebitssigmoid.py shows gradient descent with sigmoid and softmax activation functions with adam optimized for three bit xor
using tensorflow library


# running the code

one can use jupyter notebook to run this sample
pre-reqs; need python3 and numpy, matplotlib packages installed

# sigmoid and tanh functions

![image](https://github.com/bhatth2020/python/assets/148010912/85146a0b-02c8-415b-ae2a-8712bad9c645)

# derivative of sigmoid and tanh functions

![image](https://github.com/bhatth2020/python/assets/148010912/996c8e98-578d-444f-b354-a0dfdf506116)


# output of reduction in loss using sigmoid and tanh activation functions

Error at epoch with sigmoid 0: 0.4982595484906155
Error at epoch with sigmoid 1000: 0.10903310696943963
Error at epoch with sigmoid 2000: 0.0547035899360479
Error at epoch with sigmoid 3000: 0.040875044779338274
Error at epoch with sigmoid 4000: 0.03391735032534888
Error at epoch with sigmoid 5000: 0.02956541591015565
Error at epoch with sigmoid 6000: 0.026522321941849513
Error at epoch with sigmoid 7000: 0.024243951721576375
Error at epoch with sigmoid 8000: 0.022457285600249365
Error at epoch with sigmoid 9000: 0.021008486005392797
Error at epoch with sigmoid 10000: 0.019803477262718857
Error at epoch with sigmoid 11000: 0.018781106872341715
Error at epoch with sigmoid 12000: 0.017899693817196225
Error at epoch with sigmoid 13000: 0.01712973105520528
Error at epoch with sigmoid 14000: 0.01644967149180123
Error at epoch with sigmoid 15000: 0.015843368159202804
Error at epoch with sigmoid 16000: 0.015298452662524817
Error at epoch with sigmoid 17000: 0.014805271051072287
Error at epoch with sigmoid 18000: 0.014356164094620186
Error at epoch with sigmoid 19000: 0.01394496762453251
Error at epoch with sigmoid 20000: 0.013566657650906266

Error at epoch with tanh 0: 0.5061713052550134
Error at epoch with tanh 1000: 0.1363625069013134
Error at epoch with tanh 2000: 0.13865803134615828
Error at epoch with tanh 3000: 0.1387188273872178
Error at epoch with tanh 4000: 0.13859062127149951
Error at epoch with tanh 5000: 0.1384973626854209
Error at epoch with tanh 6000: 0.13843149793455645
Error at epoch with tanh 7000: 0.13838324183577885
Error at epoch with tanh 8000: 0.1383465476250198
Error at epoch with tanh 9000: 0.13831776105002466
Error at epoch with tanh 10000: 0.13829459463258034
Error at epoch with tanh 11000: 0.13827555660996887
Error at epoch with tanh 12000: 0.13825963685356732
Error at epoch with tanh 13000: 0.1382461285394989
Error at epoch with tanh 14000: 0.13823452286417562
Error at epoch with tanh 15000: 0.13822444449432203
Error at epoch with tanh 16000: 0.138215610609376
Error at epoch with tanh 17000: 0.13820780410521621
Error at epoch with tanh 18000: 0.13820085557951717
Error at epoch with tanh 19000: 0.138194630924577
Error at epoch with tanh 20000: 0.1381890225960804

#plotting gradient descent with sigmoid activation function
![image](https://github.com/bhatth2020/python/assets/148010912/4e2c6ab8-2d5d-49e6-ade4-94c000676a82)

#plotting gradient descent with tanh activation function
![image](https://github.com/bhatth2020/python/assets/148010912/9685d313-743c-4b93-bdc9-8785b7a5670a)

#plotting gradient descent with bias with sigmoid activation function
![image](https://github.com/bhatth2020/python/assets/148010912/54c3c26f-c1e1-41dc-890e-140088681755)


#plotting gradient descent with bias with tanh activation function
![image](https://github.com/bhatth2020/python/assets/148010912/bcd97002-e25c-4d1f-8e38-a4cbcebd21c6)

#plotting gradient descent with relu and softmax activation functions with adam optimizer for three bit xor
The code is in xorwiththreebitsactivation.py, it uses tensorflow library
3-bit Predictions (Full):
[[0.9924 0.0076]
 [0.0024 0.9976]
 [0.0114 0.9886]
 [0.9934 0.0066]
 [0.0086 0.9914]
 [0.9964 0.0036]
 [0.9872 0.0128]
 [0.008  0.992 ]]

![image](https://github.com/bhatth2020/python/assets/148010912/c51857d1-eac8-4a9e-9876-781aed663ded)


#plotting gradient descent with sigmoid and softmax activation functions with adam optimized for three bit xor
3-bit Predictions (Full):
[[0.9108 0.0892]
 [0.0811 0.9189]
 [0.0882 0.9118]
 [0.8878 0.1122]
 [0.1105 0.8895]
 [0.934  0.066 ]
 [0.9376 0.0624]
 [0.0601 0.9399]]

![image](https://github.com/bhatth2020/python/assets/148010912/1403c661-db42-4253-8a56-326a121b42b7)


#plotting gradient descent with gelu and softmax activation functions with adam optimizer for three bit xor


This training takes more compute time as gelu is computationally more complex
3-bit Predictions (Full):
[[0.9924 0.0076]
 [0.0024 0.9976]
 [0.0114 0.9886]
 [0.9934 0.0066]
 [0.0086 0.9914]
 [0.9964 0.0036]
 [0.9872 0.0128]
 [0.008  0.992 ]]

![image](https://github.com/bhatth2020/python/assets/148010912/316a597b-e140-4387-95a7-e05fbc5e7854)


#plotting gradient descent with tanh and softmax activation functions with adam optimizer for three bit xor

3-bit Predictions (Full):
[[0.9972 0.0028]
 [0.0014 0.9986]
 [0.0012 0.9988]
 [0.9981 0.0019]
 [0.002  0.998 ]
 [0.9993 0.0007]
 [0.999  0.001 ]
 [0.0016 0.9984]]

![image](https://github.com/bhatth2020/python/assets/148010912/7c568f59-88f6-4460-a6ff-a58daa1741e9)


#plotting gradient descent with selu, softmax activation and adam optimizer for three bit xor

3-bit Predictions (Full):
[[0.9875 0.0125]
 [0.0088 0.9912]
 [0.0211 0.9789]
 [0.9878 0.0122]
 [0.0179 0.9821]
 [0.9849 0.0151]
 [0.9744 0.0256]
 [0.0178 0.9822]]

![image](https://github.com/bhatth2020/python/assets/148010912/d1d4226c-448c-4b75-99d9-6a38ae463739)


