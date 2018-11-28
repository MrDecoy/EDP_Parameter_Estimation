import numpy as np
import math
import random
import pprint
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class Heat:

    def __init__(self, inputs):
        """
        Initiates a heat equations given all the inputs for the problem and the solver.
        :param dict inputs: all input values for


        """
        self.x = None  # x values for grid
        self.t = None  # t values for grid
        self.R = None  # Perturbation values
        self.uR = None  # solution with disturbance
        self.first = True  # place-holder variable for plotting
        self.uE = None  # Exact solution for the problem
        self.h = None  # step in x
        self.k = None  # step in t
        self.step_x = inputs['step_x']

        self.step_t = inputs['step_t']
        self.min_t = inputs['min_t']
        self.max_t = inputs['max_t']  # change this to get the desired t
        self.min_x = inputs['min_x']
        self.max_x = inputs['max_x']
        self.epsilon = inputs['pertubation_epsilon']

    def f(self, t, k):
        '''
        f is the source function for the heat equation
        :param float t: Single value for t.
        :param list k: Parameters of the function f. Should be a list of 2 values k = [k1,k2]
        '''
        return k[0] * math.exp(-k[1] * t)

    def g(self, x):
        """
        g is the function such that u(0,x) = g(x)
        :param x: point x in which we will calculate x
        :return: value of g calculated in point x
        """
        return x * (1 - x)

    def makeGrid(self):
        """
        Generates a grid of size (minT, maxT) x (minX, maxX) in R^2 x R^2, with step size of (stepT, stepX).
        """
        self.h = self.step_x
        self.k = self.step_t
        self.t, self.x = np.meshgrid(np.arange(self.min_t, self.max_t, self.step_t), np.arange(self.min_x, self.max_x
                                                                                               , self.step_x))

    def getPerturbation(self):
        '''
        Create perturbation to the exact solution for k = [1/2, 1/2]
        :return: None
        '''
        self.R = np.zeros(self.x.shape)
        self.uR = np.zeros(self.x.shape)
        for i in range(0, len(self.t[0])):
            for j in range(0, len(self.x)):
                self.R[j, i] = random.uniform(-self.epsilon, self.epsilon)
                self.uR[j, i] = self.uE[j, i] + self.R[j, i]

    def solver(self, theta):
        """
        This solver utilizes a Crank Nicolson solver with Crout factorization.
        This is a slow method for finding the solution and an implementation of SOR should work best.
        :param list theta: Parameters for the heat equation source function
        :return np.array uK: is the solution for the given heat equation.
        :return np.array errorCurve: point-wise error.
        :return float error: Quadratic error compared to uR.
        """

        m = len(self.x)
        n = len(self.t[0])
        h = self.step_x
        k = self.step_t
        lamb = k / (h * h)
        w = np.zeros(m + 1)
        l = np.zeros(m + 1)
        u = np.zeros(m + 1)

        uK = np.zeros(self.x.shape)
        print('comecei para k= ({},{})'.format(theta[0], theta[1]))
        startTime = time.clock()
        if self.first:
            self.uR = np.zeros(self.x.shape)
            self.first = False
        error = 0
        errorCurve = np.zeros(self.x.shape)
        z = np.zeros(m + 1)
        w[m] = 0  # following the initial condition u(0,t) = u(l,t) = 0. If needed, change this.
        for i in range(1, m - 1):
            w[i] = self.g(i * h)

        l[1] = 1 + lamb
        u[1] = -lamb / (2 * l[1])
        for i in range(2, m - 1):
            l[i] = 1 + lamb + lamb * u[i - 1] / 2
            u[i] = -lamb / (2 * l[i])

        l[m - 1] = 1 + lamb + lamb * u[m - 2] / 2
        for j in range(1, n + 1):
            t = j * k  # current t
            z[1] = ((1 - lamb) * w[1] + lamb / 2 * w[2] + self.f(t, theta)) / l[1]
            for i in range(2, m):
                z[i] = ((1 - lamb) * w[i] + lamb / 2 * (w[i + 1] + w[i - 1] + z[i - 1]) + self.f(t, theta)) / l[i]
            w[m - 1] = z[m - 1]
            for i in range(m - 2, 0, -1):
                w[i] = z[i] - u[i] * w[i + 1]

            for i in range(0, m + 1):
                x = i * h
                # print(x, w[i])
                # print('oi')
                uK[i - 1, j - 1] = w[i]
                self.t[i - 1, j - 1] = t
                self.x[i - 1, j - 1] = x
                error += pow(w[i] - self.uR[i - 1, j - 1], 2) / uK.size
                errorCurve[i - 1, j - 1] = (pow(w[i] - self.uR[i - 1, j - 1], 2)) / uK.size
        print('acabei para k= ({},{}) em {} segundos'.format(theta[0], theta[1], time.clock() - startTime))
        return (uK, error, errorCurve, theta)

    def fastSolver(self,theta):
        '''
        Considere o problema Aw = Bw^{-1} + f onde
        diagonal de A = a, diagonal superior = c, diag inferior = b

        :param theta:
        :return:
        '''
        m = len(self.x)
        n = len(self.t[0])
        h = self.step_x
        k = self.step_t
        lamb = k / (h * h)
        w = np.zeros(m)
        l = np.zeros(m )
        diagonalsA = [1+lamb, -lamb/2, -lamb/2]
        diagonalsB = [1-lamb, lamb/2, lamb/2]
        A = diags(diagonalsA, [0,-1, 1], shape=(m,m))
        a = np.full((m,), -lamb/2)
        b = np.full((m,), 1+lamb)
        c = np.full((m,), -lamb/2)

        B = diags(diagonalsB, [0,-1,1], shape=(m, m))
        u = np.zeros(m)
        f = np.zeros(m)
        d = np.zeros(m)
        u = np.zeros(self.x.shape)
        print('comecei para k= ({},{})'.format(theta[0], theta[1]))
        startTime = time.clock()
        for i in range(0, m - 1):
            w[i] = self.g(i * h)
        for i in range(0, m):
            f[i] = self.f(i * k, theta)
        w[m-1] = 0

        if self.first:
            self.uR = np.zeros(self.x.shape)
            self.first = False
        error = 0

        #This is the time loop
        for j in range(0, n):
            t = j*k
            d = B.dot(w) + f
            c[0] = c[0] / b[0]
            d[0] = d[0] / b[0]
            for i in range(1, m - 1):
                c[i] = c[i] / (b[i] - a[i] * c[i - 1])
            for i in range(1, m):
                d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1])
            w[m-1] = d[m-1]
            for i in range(m - 2, 0, -1):
                w[i] = d[i] - c[i] * w[i+1]
            d = w
            return w

        error = sum(sum(2**(u - self.uR)/u.size))
        print('acabei para k= ({},{}) em {} segundos'.format(theta[0], theta[1], time.clock() - startTime))

        return u, error, None, theta

    def plot(self, u, labels):
        '''

        :param u: Curve to plot
        :param labels: Labels for the Axis. Should be (x, y, z) labels.
        :return:
        '''
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(self.x, self.t, u)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.view_init(30, 30)
        plt.show()


def paralizedRun(inputs):
    heat = Heat(inputs)
    heat.makeGrid()
    heat.uE, error, errorCurve, k = heat.solver(theta=[1 / 2, 1 / 2])
    heat.plot(heat.uE, ['x', 't', 'Exact Solution'])
    heat.getPerturbation()
    minError = math.inf
    uF = None
    heat.plot(heat.uR, ['x', 't', 'Disturbed Solution'])
    tSize = 10
    p = Pool(tSize)
    for i in range(0, tSize):
        startTimeK = time.clock()
        k1 = np.random.sample(1)
        param = []
        for j in range(0, tSize):
            param.append([k1, np.random.sample(1)])

        results = p.map(heat.solver, param)
        for result in results:
            uK, error, errorCurve, k = result
            if error < minError:
                minError = error
                minK = k
                uF = uK
                minErrorCurve = errorCurve
        print('min error: {} for k=({}, {})'.format(minError, minK[0], minK[1]))

        print('done for k={} in {} seconds'.format(k1, time.clock() - startTimeK))
        print('--------------------------------------')
        return uF, error


def normalRun(inputs):
    heat = Heat(inputs)
    heat.makeGrid()
    heat.uE, error, errorCurve, k = heat.solver(theta=[1 / 2, 1 / 2])
    heat.plot(heat.uE, ['x', 't', 'Exact Solution'])
    heat.getPerturbation()
    minError = math.inf
    uF = None
    heat.plot(heat.uR, ['x', 't', 'Disturbed Solution'])
    tSize = 100
    for i in range(0, tSize):
        print('--------------------------------------')
        for j in range(0, tSize):
            k1 = np.random.sample(1)
            k2 = np.random.sample(1)
            uK, error, errorCurve, k = heat.solver([k1, k2])
            print("E: {} mE: {}".format(error, minError))
            if error < minError:
                uF = uK
                minError = error
                minK = (k1, k2)
                minCurve = errorCurve
                # heat.plot(errorCurve, ['x', 't', 'Error'])
                # heat.plot(uK, ['x', 't', 'Solution w/ k=({},{})'.format(k1,k2)])
        heat.plot(uF, ['x', 't', 'Solution w/ k=({},{})'.format(minK[0], minK[1])])

    print("new minima: {}, {}".format(k1, k2))
    heat.plot(uF, ['x', 't', 'Solution w/ k=({},{})'.format(minK[0], minK[1])])
    return uF, error, minK



if __name__ == '__main__':
    # TODO transform inputs into a xml or JSON file.
    inputs = {
        'step_x': 0.001,
        'step_t': 0.001,
        'min_t': 0,
        'max_t': 5,
        'min_x': 0,
        'max_x': 1,
        'pertubation_epsilon': 0.1,
    }

    uF, error, minK = normalRun(inputs)


