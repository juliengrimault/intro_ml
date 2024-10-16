import numpy as np
import hw02

## Part 1
original_data = np.array([
    [200, 800, 200, 800],
    [0.2,  0.2,  0.8,  0.8]
])
original_th = hw02.cv([0, 1])
th0= -0.5

n = np.shape(original_data)[1]

labels = np.array([[-1, -1, 1, 1]])

def margin_separator_through_origin(th, data, labels):
    norm_th = np.linalg.norm(th)
    errors = np.dot(th.T, data) * labels / norm_th
    return np.min(errors)

def calculate_mistakes(th, data, labels):
       gamma = margin_separator_through_origin(th, data, labels)
       bigR = np.max(np.apply_along_axis(np.linalg.norm, 0, data))
       maxError = (bigR / gamma) ** 2
       print("Gamma= {}, R= {}, (R/gamma)^ 2 = {}".format(gamma, bigR, maxError))

def run_perpeptron(data, labels, T=10000):
    def print_mistake(out={}):
        def F(pair):
            if out['counter'] != None:
                out['counter'] += 1
        return F
    
    result = {'counter': 0}
    hw02.perceptron(data, labels, params={'T': T} , hook=print_mistake(result))
    print("Run perceptron {} times, mistakes_count= {}".format(T, result['counter']))
     
data = np.vstack([original_data, np.ones(n)])
th= np.append(original_th, th0)
calculate_mistakes(th, data, labels)

scale = 0.001
th2 = np.append(original_th, th0 * scale)
data2 = np.vstack([original_data * scale, np.ones(n)])
calculate_mistakes(th2, data2, labels)


data3 = np.concatenate([original_data[:1] * scale, original_data[1:], np.ones((1, n))], axis=0)
calculate_mistakes(th, data3, labels)
run_perpeptron(data3, labels)


## Part 2

print("2A")
data = np.array([[2, 3,  4,  5]])
labels = np.array([[1, 1, -1, -1]])

(th, th0) = hw02.perceptron(data, labels)
print("(th = {}, th0 = {})".format(th, th0))

to_predict = np.array([[1, 6]])
print(hw02.positive(to_predict, th, th0))

print("-----")
print("2E")
def one_hot(k, n):
     result = np.zeros((1, n))
     result[0, k -1] = 1
     return result.T


data = np.concatenate([
     one_hot(2, 6),
     one_hot(3, 6),
     one_hot(4, 6),
     one_hot(5, 6),
], axis=1)
(th, th0) = hw02.perceptron(data, labels)
print("(th = {}, th0 = {})".format(th, th0))
to_predict = np.concatenate([
     one_hot(1, 6),
     one_hot(6, 6)    
], axis=1)

print(hw02.positive(to_predict, th, th0))
print(hw02.y(to_predict, th, th0) / np.linalg.norm(th))

data = np.concatenate([
     one_hot(1, 6),
     one_hot(2, 6),
     one_hot(3, 6),
     one_hot(4, 6),
     one_hot(5, 6),
     one_hot(6, 6)
], axis=1)
labels = np.array([[1, 1, -1, -1, 1, 1]])
(th, th0) = hw02.perceptron(data, labels)
print("(th = {}, th0 = {})".format(th, th0))
print(hw02.positive(to_predict, th, th0))

## Part 3
