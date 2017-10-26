import numpy as np

def get_x(y, stride, thres):
    x = np.floor((100*(100-stride*y)*(1+thres)-20000*thres)/(stride*(100-stride*y)*(1+thres))).astype(int)
    if x < 0:
        return 0
    
    return x
    
def nms_2d(score_map, stride, thres):
    '''
    Assume proposals are always 100 by 100, centered at pixel locations
    input:
        score_map: 2D per pixel scores
        stride: the stride between neighboring pixels
        thres: suppression iou threshold
    output:
        rst: list of (row,column) pixels that are preserved after nms
    '''
    
    height,width = score_map.shape
    
    radius = get_x(0, stride, thres)
    cover_mask = np.ones((2*radius+1, 2*radius+1))
    for rr in range(radius+1):
        xx = get_x(rr, stride, thres)
        for xxi in range(xx+1):
            cover_mask[radius-rr, radius-xxi] = 0
            cover_mask[radius-rr, radius+xxi] = 0
            cover_mask[radius+rr, radius-xxi] = 0
            cover_mask[radius+rr, radius+xxi] = 0
            
        
    score_map_padded = np.pad(score_map, ((radius, radius),(radius, radius)), 'constant')
    score_map_mask = np.ones_like(score_map_padded)
    
    rst = []
    while np.sum(score_map_mask[radius:radius+height, radius:radius+width]) > 0:
        sm_curr = score_map*score_map_mask[radius:radius+height, radius:radius+width]
        rr,cc = np.unravel_index(np.argmax(sm_curr), sm_curr.shape)
        rst.append((rr,cc))
        
        score_map_mask[rr:rr+2*radius+1, cc:cc+2*radius+1] = score_map_mask[rr:rr+2*radius+1, cc:cc+2*radius+1]*cover_mask
        
        
    return rst
