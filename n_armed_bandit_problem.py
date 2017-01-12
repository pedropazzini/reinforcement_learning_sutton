import numpy as np
from scipy import stats

class n_armed_bandit(object):
    '''
    Class that simulates a n-armed bandit problem. (https://en.wikipedia.org/wiki/Multi-armed_bandit)

    On this implementation, all the n armed bandits give rewards generated from normal distributions.
    '''

    def __init__(self, n = 10, vec_mu_std=None):
        '''
        Initiates the n_armed_bandit

        Parameters:
        -----------

        n: integer, > 0, default=10
        
            The number of armed bandits.

        vec_mu_std: array of n tuples (mu,std), default None

            mu is the mean of the normal distribution and std its standard deviation. If None is passed, then an array with uniform values in the intervals [0,1] for mu and [0,1] for std are generated.

        '''

        if n <=0:
            raise ValueError('\'n\' should be greather than 0')

        self.n = int(n)

        self.vec_mu_std = vec_mu_std

        if self.vec_mu_std is None:
            self.vec_mu_std = [(np.random.uniform(0,1),np.random.uniform(0,1)) for _ in range(n)]
        elif n != len(self.vec_mu_std):
            raise ValueError('Incoherent value of \'n\' and the size of \'vec_mu_std\'')

        self.vec_armed_bandits = [stats.norm(loc=i[0],scale=i[1]) for i in self.vec_mu_std]

    def get_mus(self):
        '''
        Returns the real expected value (mu) of each normal distribution
        '''

        return np.array([i[0] for i in self.vec_mu_std])

    def query_max(self):
        '''
        Querys the max mean expected reward.

        Returns:
        --------

        max_: float
            The maximum expected reward.
        '''
        max_ = -float(inf)
        for t in self.vec_mu_std:
            if t[0] > max_:
                max_ = t[0]

        return max_

    def query(self,i):
        '''
        Query the ith armed_bandit.

        Parameters:
        -----------

        i: integer > 0 and < n

            The position of the armed bandith that should be quered.

        Returns:
        --------

        sample: float

        A sample from the normal distribution of the ith armed bandith.
        '''

        if i < 0 or i >= self.n:
            raise ValueError('Wrong value for \'i\'. Should be in the interval 0 < \'i\' < \'n\'.')

        i = int(i)

        return self.vec_armed_bandits[i].rvs()

    def run_step(self):
        '''
        Runs one step of the n-armed bandith problem.

        Returns:
        --------
        rewards: np.array of floats, [n,]

            The rewards of each armed bandit.
        '''

        rewards = [self.query(i) for i in range(self.n)]
        return np.array(rewards)

    def run_steps(self,n_steps=1000):
        '''
        Run multiple steps of the n-armed bandit.

        Parameters:
        -----------
        n_steps: integer, >0

            The number of steps to be run.

        Returns:
        --------

        M: np.array, [n_steps,n]

            A matrix containig the rewards in each step
        '''

        M = np.array([self.run_step() for _ in range(n_steps)])

        return M

    def get_optimal(self,M):
        '''
        Gets the optimal action-reward in all steps of matrix M

        Parameters:
        -----------

        M: np.array [n_steps,n]

            Matrix containing the rewards in each step

        Returns:
        --------

        vec: np.array [n_steps,]

            Vector of the optimum cummulative reward after each step.
        '''

        vec = np.array([M[i,np.argmax(np.mean(M[:(i+1)],axis=0))] for i in range(M.shape[0])])

        return np.cumsum(vec)

    def get_greedy(self,M):
        '''
        Gets the greedy reward from the steps in M

        Parameters:
        -----------

        M: np.array [n_steps,n]

            Matrix containing the rewards in each step

        Returns:
        --------

        vec: np.array [n_steps,]

            Vector of th cummulative reward using the greedy strategy.
        '''

        st = [[0] for _ in range(M.shape[1])]
        dic_st = dict((i,val) for i,val in enumerate(st))
        vec = np.zeros((M.shape[0],))
        for i in range(M.shape[0]):
            max_i = max(dic_st.items(), key=operator.itemgetter(1))[0]
            choice = M[i,max_i]
            st[max_i].append(choice)
            dic_st[max_i] = np.mean(st[max_i])
            vec[i] = choice

        return np.cumsum(vec)

    def get_eta(self,M,eta=0.1):
        '''
        Get the reward from the steps in M using the eta strategy.

        Parameters:
        -----------

        M: np.array [n_steps,n]

            Matrix containing the rewards in each step

        eta: float, 0 > eta < 1

            The eta parameter from the strategy. When eta=0 it is the greedy strategy, when eta=1 is the random strategy.

        Returns:

        vec: np.array [n_steps,]

            Vector of th cummulative reward using the eta strategy.
        '''
        pos_greedy = np.argmax(M[0])

        vec = np.zeros((M.shape[0]))

        for i in range(M.shape[0]):

            if np.random.uniform() < eta:
                pos_greedy = np.argmax(np.mean(M[:i],axis=0))

            vec[i] = M[i,pos_greedy]

        return np.cumsum(vec)

    def run_multiple_times(self,n_times=2000,n_steps=1000):
        '''
        Run the steps multiple times

        Parameters:
        -----------

        n_times: integer, > 0

            The number of times that the steps should be runned.

        n_steps: integer, >0

            The number of steps

        Returns:
        --------

            The optimal and greedy rewards on the steps
        '''

        grd = np.zeros((n_times,n_steps))
        opt = np.zeros((n_times,n_steps))

        for i in range(n_times):
            M = self.run_steps(n_steps)
            grd[i] = self.get_greedy(M)
            opt[i] = self.get_optimal(M)

        return grd,opt

