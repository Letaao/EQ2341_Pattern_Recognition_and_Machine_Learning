import numpy as np
from .DiscreteD import DiscreteD


class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        # to check if it's infinite, if A is a square matrix,exit N+1 (textbook P98.99)
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True

    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q    # eye 对角矩阵，@ matrix multiplication

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t + np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence, 整数行向量，state sequence generated
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message

        S = []
        nStates = self.A.shape[0]

        # print(nStates, 'number of states(non-exit)')
        exit_state = nStates + 1

        # the initial state and its transitions conditioned on the current state
        # can be seen as discrete random variables, generated from DiscreteD

        S.append(DiscreteD(self.q).rand(1)[0])
        # print(S,'initial')
        current_state = S[0]  # temp_variable
        # print(current_state, 'current_state')

        for t in range(2, tmax + 1):
            S.append(DiscreteD(self.A[current_state - 1]).rand(1)[0])
            # print(S)
            current_state = S[t - 1]
            # print(current_state, 'current_state')
            if current_state == exit_state:
                # print('the time jump to exit is %s' % t)
                break
        # print(S)
        return S


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, p_x):  # p_x is a matrix of shape N, T
        # print(p_x)
        alpha_hat, c = [], []  # scaled forward variable, scaled factors (c)
        n_states = p_x.shape[0]  # N
        t_max = p_x.shape[1]  # T

        """initialization"""
        a_temp = [self.q[j] * p_x[j, 0] for j in range(n_states)]  # eq 5.42
        c.append(sum(a_temp))  # eq 5.43
        alpha_hat.append([np.divide(a, c[0]) for a in a_temp])  # eq 5.44

        """forward step"""
        for t in range(1, t_max):
            a_temp = []
            for j in range(n_states):
                con_prob = sum([np.multiply(alpha_hat[t-1][i], self.A[i, j]) for i in range(n_states)])  # eq 5.49
                a_temp.append(p_x[j, t] * con_prob)  # eq 5.50
            c_t = sum(a_temp)  # eq 5.51
            c.append(c_t)
            alpha_hat.append([a/c_t for a in a_temp])  # eq 5.52

        """termination"""
        if self.is_finite:
            c_tail = sum([alpha_hat[-1][i] * self.A[i, -1] for i in range(n_states)])  # eq 5.53
            c.append(c_tail)

        return alpha_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self, c, p_x):
        beta_hat = []

        """initialization"""
        if self.is_finite:
            beta_hat0 = [beta/(c[-1]*c[-2]) for beta in self.A[:, -1]]  # eq 5.65
        else:
            beta_hat0 = [1/c[-1] for i in range(self.A.shape[0])]  # eq 5.64
        beta_hat.insert(0, beta_hat0)

        """backward step"""
        c = c[:-2] if self.is_finite else c[:-1]
        c.reverse()
        for t in range(len(c)):
            beta_hat_t = []
            for i in range(self.A.shape[0]):
                beta_i_t = sum([p_x[j, -t-1]*self.A[i, j]*beta_hat[0][j] for j in range(self.A.shape[0])])
                beta_hat_i_t = beta_i_t/c[t]
                beta_hat_t.append(beta_hat_i_t)  # eq 5.70
            beta_hat.insert(0, beta_hat_t)

        return beta_hat


    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
