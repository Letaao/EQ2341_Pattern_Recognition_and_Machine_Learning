import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    state-generator component
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


        self.nStates = transition_prob.shape[0] # row

        self.is_finite = False
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
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

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
        
        logpD = np.log(aii)*t+ np.log(1-aii)
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
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs

        Parameters:
        self.q  #InitialProb(i)= P[S(1) = i]
        self.A  #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
        A Markov state sequence S(t), t=1..T
            is determined by fixed initial probabilities P[S(1)=j], and
            fixed transition probabilities P[S(t) | S(t-1)]

        """
        
        #*** Insert your own code here and remove the following error message 
        S = np.empty(tmax, dtype=int) #Markov state sequence, NOT INCLUDING the END state
        S[0] = DiscreteD(self.q).rand(1) #generate the first state according to initial Probability
        end_state = self.nStates
        current_state = S[0]

        for t in range(1,tmax):
            new_state = DiscreteD(self.A[current_state]).rand(1) # S(t)=0,..,T
            if new_state == end_state:
                S = S[0:t]
                break
            S[t] = new_state
            current_state = S[t]
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

    def forward(self,pX):
        T = pX.shape[1]
        N = self.A.shape[0]       
        c = np.zeros(T)
        temp = np.zeros((N, T))
        alfaHat = np.zeros((N, T))
        
        # Initialization
        for i in range(0, N):
            temp[i, 0] = self.q[i] * pX[i, 0]
            c[0] = c[0] + temp[i, 0]
        for i in range(0, N):
            alfaHat[i, 0] = temp[i, 0] / c[0]

        # Forward Steps
        for t in range(1, T):
            for j in range(0, N):
                temp[j, t] = pX[j, t] * np.sum(alfaHat[:, t - 1] * self.A[:, j])
                c[t] = c[t] + temp[j, t]
            for j in range(0, N):
                alfaHat[j, t] = temp[j, t] / c[t]
            
        # Termination
        if self.is_finite == True:
            tempc = c
            c = np.zeros(T + 1)
            c[:T] = tempc
            for i in range(0, N):
                c[T] = c[T] + alfaHat[i, T - 1] * self.A[i, N]

        return alfaHat, c

    def finiteDuration(self):
        pass

    def backward(self, c, pX):
        T = pX.shape[1]
        N = self.A.shape[0]
        betaHat = np.zeros((N, T))

        # Initialization
        if self.is_finite == False:
            for i in range(0, N):
                betaHat[i, T-1] = 1/c[T-1]
        else:
            for i in range(0, N):
                betaHat[i, T-1] = self.A[i, N]/(c[T]*c[T-1])

        # Backward Step
        if self.is_finite == True:
            self.A = self.A[:, :N]
            
        for t in range(T-1, 0, -1):
            for i in range(0, N):
                betaHat[i, t-1] = np.sum(self.A[i, :] * pX[:, t] * betaHat[:, t]) / c[t-1]

        return betaHat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
