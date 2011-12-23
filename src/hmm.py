"""
This is a very niave implementation of a HMM aimed to cover:
Filtering, Prediction, Smoothing, Likelihood (forward-backward) and Most Likely Explanation (viterbi)
"""

class HMM():
    
    def __init__(self,states,transition_model, sensor_model, prior):
        self.states = states    #count states
        self.transition_model = transition_model#Tij = P(Xt=j|Xt-1=i)
        self.sensor_model = sensor_model
        self.prior = prior
        #the only bits we NEED are the transition model and the sensor model
        #prior can be estimated from the transition model
        #states are implicit
        #TODO: calculate prior and states if omitted
    
    def filtering(self,evidence,pi=None):
        if pi is None:
            pi = self.prior[:]
        for ei in evidence:
            #prediction ignoreing evidence
            pi = self.prediction(pi)
            #multiply by the evidence and normalise
            pi = self.forward(ei,pi)
            #normalise
            pi = self.normalise(pi)
        return pi
    
    def normalise(self,pi):
        a = 1/sum(pi)
        pi = [x*a for x in pi]
        return pi
    
    def likelihood(self,evidence,pi=None):
        if pi is None:
            pi = self.prior[:]
        for ei in evidence:
            #prediction ignoreing evidence
            pi = self.prediction(p)
            #multiply by the evidence and normalise
            pi = self.forward(ei,pi)
        return pi
    
    def forward(self,ei,prior):
        ei_model = self.sensor_model[ei]
        pi = prior[:]
        for k in range(len(pi)): 
            pi[k] = pi[k]*ei_model[k]
        return pi
    
    def prediction(self,p0=None,t=1):
        if p0 is None:
            p0 = self.prior[:]
        pi = p0
        for i in range(t):
            pi = self.predict_onestep(pi)
        return pi
        
    def predict_onestep(self,p0):
        pi = [0]*self.states
        for k in range(len(self.transition_model)):
            sub_model = self.transition_model[k]
            for j in range(len(sub_model)):
                pi[j] = pi[j]+sub_model[j]*p0[k]
        return pi
        
    def backward(self, ei, bi1=None):
        if(bi1 is None):
            bi1 = [1]*self.states
        bi = [0]*self.states
        ei_model = self.sensor_model[ei]
        for k in range(len(self.transition_model)):
            sub_model = self.transition_model[k]
            for j in range(len(sub_model)):
                bi[j] = bi[j]+sub_model[j]*ei_model[k]*bi1[k]
        return bi

    def smoothing(self,evidence):
        #working backward through the evidence calculate a new prior
        bi1 = [1]*self.states
        bi = bi1
        for i in range(len(evidence)-1,-1,-1):
            ei = evidence[i]
            #calculate the backward message
            bi = self.backward(ei,bi1)
            #normalise
            bi = self.normalise(bi)
        return bi
    
    def smoothing_k(self,evidence,k,p0=None):
        #calculate the prior for K given evidence from 0 to t where t < k
        #We filter forward and smooth backward
        filter_evidence = evidence[:k]
        smoothing_evidence = evidence[k:]
        #filter forward
        f = self.filtering(filter_evidence,p0)
        #smooth backward
        s = self.smoothing(smoothing_evidence)
        p = [0]*self.states
        #multiply the two together
        for i in range(len(f)):
            p[i] = f[i]*s[i]
        p = self.normalise(p)
        return p
        
    #forward backward
    def forward_backward(self,evidence,p0=None):
        #setup
        t = len(evidence)+1
        fv = [0]*t
        sv = [0]*t
        b = [1]*self.states
        if not p0 is None:
            fv[0] = p0[:]
        else:
            fv[0] = self.prior[:]
        sv[0]=fv[0]
        #forward messages
        for i in range(1,t):
            ei = evidence[i-1]
            fv[i] = self.forward(ei,fv[i-1])
        for i in range(t-1,0,-1):
            ei = evidence[i-1]
            #fv[i] x b
            fb = [fv[i][j]*b[j] for j in range(len(fv[i]))]
            sv[i] = self.normalise(fb)
            b = self.backward(ei,b)
        return sv
            
    def viterbi(self,evidence,pi=None):
        v = [[]]
        v[0] = self.filtering([evidence[0]],pi)
        path = {}
        for i in range(self.states):
            path[i]=[i]
        #for each time step
        for t in range(1,len(evidence)):
            v.append([])
            newpath = {}
            et = evidence[t]
            et_model = self.sensor_model[et]
            v1 = v[t-1]
            #for each possible state
            for i in range(self.states):
                #choose the state that maximises the forward message
                (prob,state) = max([(v1[s0]*self.transition_model[s0][i]*et_model[i],s0) for s0 in range(self.states)])
                v[t].append(prob)
                newpath[i]=path[state][:]
                newpath[i].append(i)
            path = newpath
        #choose the most likely
        (prob,state) = max([(v[-1][s0], s0) for s0 in range(self.states)])
        return prob,path[state]

states = 2
transition_model = [[0.7,0.3],[0.3,0.7]] 
sensor_model = [[0.8,0.1],[0.2,0.9]]
prior = [0.5,0.5]
h = HMM(states,transition_model,sensor_model,prior)
#p = h.filtering([1,1])
#print p
#p = h.likelihood([1,1,1])
#print p
#print h.prediction([0.1,0.9],t=50)
#print h.smoothing([1])
#print h.smoothing_k([1,1],1)
#print h.forward_backward([1,1])
#print h.forward_backward([1,1,1])
print h.viterbi([1,1,0,1,1])