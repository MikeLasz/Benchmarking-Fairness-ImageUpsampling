import numpy as np 


# Divergence measures between prob and the uniform distribution 
def kl(prob):
    distance = 0 
    for p in prob:
        distance += p * np.log(p) 
    return distance + np.log(len(prob))

def cheb(prob):
    distance = np.abs(prob - 1/len(prob)) 
    return np.max(distance)

def tv(prob):
    distance = np.abs(prob - 1/len(prob))
    return np.sum(distance)

def chi(prob):
    distance = (prob - 1/len(prob))**2 
    return len(prob) * np.sum(distance)


# Computing P_{RDP} and P_{PR}
def compute_RDP(dfs, setting, methods, races):
    rdps = {}
    for method in methods:
        df = dfs[setting][method]
        prob = [] 
        for race in races:
            prob.append(np.mean(df["race_0-1"][df["race"]==race]))
        rdps[method] = prob / np.sum(prob) 
    return rdps 

def compute_PR(dfs, setting, methods, races): 
    prs = {}
    for method in methods:
        df = dfs[setting][method]
        prob = [] 
        for race in races:
            prob.append(len(df.loc[df["race_recon"]==race]))
        prs[method] = prob / np.sum(prob) 
    return prs 

def compute_UCPR(dfs, setting, methods, races): 
    ucprs = {}
    for method in methods:
        df = dfs[setting][method]
        prob = [] 
        for race in races:
            prob.append(len(df.loc[df["race_recon"]==race]))
        ucprs[method] = prob / np.sum(prob) 
    return ucprs 