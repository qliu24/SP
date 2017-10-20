import numpy as np

def nms(boxes, overlap):
    assert(boxes.shape[1]==5)
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1)*(y2-y1+1)
    
    I = np.argsort(s)
    pick = []
    while I.size!=0:
        i = I[-1]
        pick.append(i)
        suppress = [I.size-1]
        for pos in range(I.size-1):
            j = I[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            w = xx2-xx1+1
            h = yy2-yy1+1
            if w>0 and h>0:
                # o = w*h/area[j]
                o = w*h/(area[i]+area[j]-w*h)
                if o > overlap:
                    suppress.append(pos)
        
        I = np.delete(I, suppress)
        
    return pick