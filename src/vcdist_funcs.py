import numpy as np

def vc_dis_paral(inpar):
    inst_ls, idx = inpar
    rst = np.ones(len(inst_ls))
    for nn in range(idx+1, len(inst_ls)):
        rst[nn] = (vc_dis(inst_ls[idx], inst_ls[nn])+vc_dis(inst_ls[nn], inst_ls[idx]))/2
        
    return rst

    
def vc_dis(inst1, inst2):
    inst1_padded = np.pad(inst1, ((1,1),(1,1),(0,0)), 'constant')
    
    dis_cnt_deform = 0
    where_f = np.where(inst2==1)
    if len(where_f[0])==0:
        return 0.0
    
    for nn1, nn2, nn3 in zip(where_f[0], where_f[1], where_f[2]):
        if inst1_padded[nn1:nn1+3, nn2:nn2+3, nn3].sum()==0:
            dis_cnt_deform += 1
    
    return float(dis_cnt_deform)/len(where_f[0])