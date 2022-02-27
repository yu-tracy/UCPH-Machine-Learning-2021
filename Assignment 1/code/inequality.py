import numpy as np
import matplotlib.pyplot as plt
import math

def draw_plot():
    # Bernoulli random variables X1,...,X20 (20 coins) with bias 1/2 and α ∈{0.5,0.55,0.6,...,0.95,1}.

    coin_sets = np.random.randint(2, size=(10 ** 6, 20))  
    # Mean number of outcome (0/1) in each of the experiments
    coin_mean = np.mean(coin_sets, axis = 1) 
    empirical_frequency_sets = []
    value_of_alpha = []
    for i in range(50, 105, 5):
        alpha = float(i) / 100  # Alpha
        empirical_frequency = (coin_mean >= alpha).sum() / float(1000000)  # the empirical frequency
        empirical_frequency_sets.append(empirical_frequency)
        value_of_alpha.append(alpha)
    plt.subplot(2, 1, 1)
    plt.plot(value_of_alpha, empirical_frequency_sets, label = 'empirical frequency')
    plt.xlabel("alpha", loc='right')
    plt.ylabel("frequency")
    plt.title("Question 2.a with bias = 0.5")

    # compute the Markov’s bound
    bias = 0.5
    value_of_alpha = np.asarray(value_of_alpha)
    markov_bound = bias / value_of_alpha
    plt.plot(value_of_alpha, markov_bound, label = 'Markov\'s bound')

    # compute the Chebyshev’s bound
    variance = 0.0125  # 1/80
    epsilon = value_of_alpha - bias
    epsilon_sqr = np.square(epsilon)
    try:
        cheby_bound = variance / epsilon_sqr
    except:   # when epsilon_sqr = 0
        cheby_bound = 1
    np.asarray(cheby_bound)[cheby_bound > 1] = 1
    plt.plot(value_of_alpha, cheby_bound, label='Chebyshev\'s bound')

    # compute the Hoeffding’s bound
    powers = (-2) * 20 * epsilon_sqr
    hoeffding_bound = []
    for power in powers:
        hoeffding_bound.append(math.exp(power))
    plt.plot(value_of_alpha, hoeffding_bound, label = 'Hoeffding’s bound')
    plt.legend(loc='upper right')

    #Bernoulli random variables X1,...,X20 (20 coins) with bias 0.1 and α ∈ {0.1,0.15,...,1}.
    coin_sets = np.random.randint(2, size=(10 ** 6, 20))  
    # Mean number of outcome (0/1) in each of the experiments
    coin_mean = np.mean(coin_sets, axis = 1)
    empirical_frequency_sets = []
    value_of_alpha = []
    for i in range(10, 105, 5): 
        alpha = float(i) / 100  
        empirical_frequency = (coin_mean >= alpha).sum() / float(1000000)  # the empirical frequency
        empirical_frequency_sets.append(empirical_frequency)
        value_of_alpha.append(alpha)

    plt.subplot(2, 1, 2)
    plt.plot(value_of_alpha, empirical_frequency_sets, label = 'empirical frequency')
    plt.xlabel("alpha", loc='right')
    plt.ylabel("frequency")
    plt.title("Question 2.b with bias = 0.1")

    # compute the Markov’s bound
    bias = 0.1
    value_of_alpha = np.asarray(value_of_alpha)
    markov_bound = bias / value_of_alpha
    plt.plot(value_of_alpha, markov_bound, label = 'Markov\'s bound')

    # compute the Chebyshev’s bound
    variance = 0.0045  # 9/2000
    epsilon = value_of_alpha - bias
    epsilon_sqr = np.square(epsilon)
    try:
        cheby_bound = variance / epsilon_sqr
    except:
        cheby_bound = 1
    np.asarray(cheby_bound)[cheby_bound > 1] = 1
    plt.plot(value_of_alpha, cheby_bound, label='Chebyshev\'s bound')

    # compute the Hoeffding’s bound
    powers = (-2) * 20 * epsilon_sqr
    hoeffding_bound = []
    for power in powers:
        hoeffding_bound.append(math.exp(power))
    plt.plot(value_of_alpha, hoeffding_bound, label = 'Hoeffding’s bound')
    plt.legend(loc='upper right')
    plt.savefig('inequality.png')
    plt.close()