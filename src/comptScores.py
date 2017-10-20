import numpy as np

def comptScores(patch_feat, sp_models):
    sp_num = len(sp_models)
    
    weights = np.array([sp_models[pp][0] for pp in range(sp_num)])
    logZs = np.array([sp_models[pp][1] for pp in range(sp_num)])
    priors = np.array([sp_models[pp][2] for pp in range(sp_num)])
    
    scores = np.dot(weights,patch_feat.reshape((-1,1))).squeeze() - logZs + priors
    # scores = np.dot(weights,patch_feat.reshape((-1,1))).squeeze()
    return scores
