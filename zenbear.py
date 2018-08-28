from copy import deepcopy
import numpy as np
import pandas as pd
import math
import time
import datetime


import logging
logger = logging.getLogger('bear')

if not logger.handlers:
    hdlr = logging.FileHandler('C:/logs/zenbear.log',mode='w')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)

logger.debug('START')


global bestpath  
global depth  
global record  
global bestlength
global returnto

bestpathcount = 0
bestpath  = ''
depth = 0
record = 10000
returnto = -1

#[0]end1, [1]end2, [2]used, [3]segmentid, [4]length
master_segments = np.array([
     [1,4,0,0,22],[1,2,0,1,22],[1,3,0,2,17],[4,3,0,3,29],[5,4,0,4,13]
    ,[5,3,0,5,16],[3,2,0,6,7]
    ,[53,54,0,7,10],[63,54,0,8,7],[5,8,0,9,27],[3,8,0,10,39]
    ,[8,16,0,11,38],[2,16,0,12,60],[2,6,0,13,43],[6,16,0,14,20],[6,7,0,15,6]
    ,[7,16,0,16,20],[7,17,0,17,35],[17,18,0,18,7],[18,16,0,19,24],[8,14,0,20,20],[14,16,0,21,43]
    ,[8,9,0,22,38],[9,11,0,23,11],[9,10,0,24,36],[10,11,0,25,35],[10,12,0,26,19]
    ,[12,11,0,27,20],[12,15,0,28,19],[15,13,0,29,22],[11,13,0,30,24],[13,8,0,31,17]
    ,[14,15,0,32,22],[15,19,0,33,38],[14,19,0,34,40],[19,16,0,35,61],[19,18,0,36,62]
    ,[19,21,0,37,42],[21,18,0,38,45],[21,22,0,39,15],[22,18,0,40,45],[22,17,0,41,50]
    ,[22,24,0,42,68],[22,23,0,43,33],[21,23,0,44,28],[23,25,0,45,44],[21,25,0,46,61]
    ,[19,25,0,47,44],[20,25,0,48,47],[19,20,0,49,23],[15,20,0,50,30],[15,27,0,51,55]
    ,[27,26,0,52,8],[26,20,0,53,34],[26,25,0,54,38],[25,24,0,55,31],[23,24,0,56,36]
    ,[24,51,0,57,13],[51,50,0,58,16],[50,25,0,59,29],[25,45,0,60,60],[45,50,0,61,41]
    ,[25,35,0,62,24],[27,28,0,63,13],[28,29,0,64,31],[29,31,0,65,9],[29,30,0,66,27]
    ,[30,31,0,67,21],[30,32,0,68,16],[32,31,0,69,10],[32,33,0,70,16],[33,34,0,71,16]
    ,[34,28,0,72,16],[34,26,0,73,10],[34,35,0,74,32],[34,38,0,75,42],[33,40,0,76,32]
    ,[40,41,0,77,7],[41,43,0,78,16],[41,42,0,79,19],[42,43,0,80,9],[42,39,0,81,25]
    ,[43,38,0,82,21],[39,38,0,83,10],[39,37,0,84,19],[36,37,0,85,6],[38,36,0,86,16]
    ,[36,35,0,87,32],[37,44,0,88,16],[44,35,0,89,31],[44,46,0,90,22],[40,43,0,91,13]
    ,[46,47,0,92,6],[47,44,0,93,20],[44,45,0,94,29],[45,35,0,95,46],[47,45,0,96,17]
    ,[51,53,0,97,55],[52,53,0,98,4],[50,52,0,99,44],[52,64,0,100,11],[45,64,0,101,19]
    ,[47,49,0,102,19],[46,48,0,103,31],[49,48,0,104,16],[49,57,0,105,18],[48,56,0,106,11]
    ,[48,55,0,107,12],[55,56,0,108,5],[56,57,0,109,13],[55,58,0,110,16],[58,57,0,111,5]
    ,[57,64,0,112,20],[58,59,0,113,10],[57,60,0,114,11],[59,60,0,115,5],[60,64,0,116,17]
    ,[64,62,0,117,10],[60,62,0,118,18],[59,61,0,119,22],[61,62,0,120,4],[61,63,0,121,7]
    ,[31,28,0,122,28],[54,62,0,123,11]
])

master_segs=pd.DataFrame(master_segments, columns=['end1','end2','used','ID','length']) 
del master_segs['ID']

front = np.array([2,6,7,17,22,24,51,53,54,63,61,59,58,55])
tail = np.array([55,48,46,44,37,39,42,41,40,33,32,30,29,28,27,15,12,10,9,8,5,4,1,2])
i = 0
# preset the known front path
#while i < len(front)-1:
#    for index, row in master_segs.iterrows():
#        if ((row['end1'] == front[i] and row['end2'] == front[i+1]) or (row['end2'] == front[i] and row['end1'] == front[i+1])) :
#            row['used'] =1
#    i +=1
    
# preset the known tail path
i=0
#while i < len(tail)-1:
#    for index, row in master_segs.iterrows():
#        if ((row['end1'] == tail[i] and row['end2'] == tail[i+1]) or (row['end2'] == tail[i] and row['end1'] == tail[i+1])) :
#            row['used'] =1
#    i +=1


logger.debug(master_segs)
     
loc = np.array([[0,0,0]
     ,[1,10,120]
     ,[2,0,101]
     ,[3,8,104]
     ,[4,35,120]
     ,[5,24,105]
     ,[6,15,60]
     ,[7,11,55]
     ,[8,41,84]
     ,[9,51,120]
     ,[10,87,120]     
     ,[11,54,109]
     ,[12,74,107]
     ,[13,58,86]
     ,[14,61,79]     
     ,[15,80,88]
     ,[16,30,49]
     ,[17,29,25]
     ,[18,36,26]
     ,[19,92,52]
     ,[20,106,73]
     ,[21,80,14]
     ,[22,72,0]
     ,[23,105,8]
     ,[24,141,5]
     ,[25,136,36]
     ,[26,140,74]
     ,[27,130,79]
     ,[28,124,90]
     ,[29,132,120]
     ,[30,165,120]
     ,[31,144,115]
     ,[32,153,110]
     ,[33,155,94]
     ,[34,142,79]     
     ,[35,152,52]     
     ,[36,174,75]     
     ,[37,179,76]     
     ,[38,183,86]     
     ,[39,195,88]          
     ,[40,183,109]     
     ,[41,182,117]     
     ,[42,200,113]     
     ,[43,194,105]     
     ,[44,181,55]     
     ,[45,195,34]          
     ,[46,205,55]     
     ,[47,200,52]     
     ,[48,235,55]     
     ,[49,219,52]     
     ,[50,164,9]     
     ,[51,152,0]          
     ,[52,207,16]     
     ,[53,206,11]     
     ,[54,210,2]     
     ,[55,242,48]     
     ,[56,237,46]     
     ,[57,226,37]          
     ,[58,233,38]     
     ,[59,233,27]     
     ,[60,227,27]     
     ,[61,217,14]     
     ,[62,213,15]     
     ,[63,215,5]          
     ,[64,211,25]     
     
])
    
logger.debug('LOADED DATA')    
    
node_loc=pd.DataFrame(loc, columns=['ID','x','y']) 
del node_loc['ID']

best_segments = master_segs

# get center point of a segment
def getcenter(x1,y1,x2,y2):
    return [(x1+x2)/2,(y1+y2)/2]

def getNodesCenter(node1,node2):
    x1 = node_loc.iloc[node1]['x']
    y1 = node_loc.iloc[node1]['y']
    x2 = node_loc.iloc[node2]['x']
    y2 = node_loc.iloc[node2]['y'] 
    return [(x1+x2)/2,(y1+y2)/2]

# get length between two points
def getlength(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2) + ((y2-y1)**2))

def getdistance(node1,node2,myposition):
    x1 = node_loc.iloc[node1]['x']
    y1 = node_loc.iloc[node1]['y']
    x2 = node_loc.iloc[node2]['x']
    y2 = node_loc.iloc[node2]['y'] 
    centerx = getcenter(x1,y1,x2,y2)[0]
    centery = getcenter(x1,y1,x2,y2)[1]
    positionx = node_loc.iloc[myposition]['x']
    positiony = node_loc.iloc[myposition]['y']
    return getlength(centerx,centery,positionx,positiony)

def getnewnodedistance(node1,node2,mynode,centerx,centery):
    if mynode == node1:
        targetnode =  node2
    else:
        targetnode =  node1
    return getlength(node_loc.iloc[targetnode]['x'],node_loc.iloc[targetnode]['y'],centerx,centery) 
    
# big mama
def iterate(node,segments,path,length):  
    
    global bestsegments
    global bestpath  
    global depth 
    global record
    global returnto
    
    #drawbear(segments)
    
    path = path + '->' + str(node).zfill(2)
    depth = path.count('->') -1
    
    # arived at right depth, stop skipping
    if returnto == depth:
        returnto = -1
    #skip if I am deeper than returnto level
    if returnto != -1 and depth > returnto:
        return
    
    prefix = 'D:'+str(depth)+'|'
    unused_segs = segments[segments.used ==0] 
    
    if len(unused_segs.axes[0]) == 0:
        logger.debug(str(datetime.datetime.now()))
        logger.debug(prefix+'Roooarr ! Bear complete!')
        # you can't possibly find a better solution from
        # the node before a completion
        returnto = depth - 1
        
        if length < record:
            record = length
            bestpath = path
            bestsegments = segments
            logger.debug(prefix+'New minimum length:'+str(record))
            logger.debug(prefix+'Best Path:'+bestpath)
        return    
    
    
    # finding the closed unused segment
    unused_segs['distance'] = unused_segs.apply(lambda row: getdistance(row['end1'], row['end2'],node), axis=1)
    unused_segs = unused_segs.sort_values(by='distance',ascending = True)

    closestUnusedSegNode1 = 0
    closestUnusedSegNode2 = 0
    closestx = 0.0
    closesty = 0.0
    closestUnusedSegNode1 = unused_segs['end1'].iloc[0]
    closestUnusedSegNode2 = unused_segs['end2'].iloc[0]
    
    closestx = getNodesCenter(closestUnusedSegNode1,closestUnusedSegNode2)[0]
    closesty = getNodesCenter(closestUnusedSegNode1,closestUnusedSegNode2)[1]

    if length >= record:
        return

    #find all connected unused segments that still keep us under max length
    distance = record - length
    conn_segs = segments[(segments['end1'] == node) | (segments['end2'] == node)]
    conn_segs = conn_segs[(conn_segs['used']< 2) & (conn_segs['length'] < distance) ]
    
    if len(conn_segs.index) == 0:
        #logger.debug(prefix+'No more connections')
        return
    # check distance to closest unused seg from each new end point    
    conn_segs['newnodedistance'] = conn_segs.apply(lambda row: getnewnodedistance(row['end1'], row['end2'],node,closestx,closesty), axis=1)   
    # create the weighting
    conn_segs['sort_val'] = (conn_segs['used']+0.1) * conn_segs['length'] * conn_segs['newnodedistance']
    conn_segs = conn_segs.sort_values(by='sort_val',ascending = True)
    
     
    #main "for loop" for iteration
    for index, row in conn_segs.iterrows():
        #print(segment)

        next_node = int(row['end1'])
        # flip if reversed order
        if next_node == node:
            next_node = int(row['end2'])

        #deepcopy path and segments
        thispath = deepcopy(path)
        thissegments = deepcopy(segments)
        thislength =  deepcopy(length)
        thissegments.iloc[index]['used'] += 1
        thislength = thislength + row['length']
        iterate(next_node,thissegments,thispath,thislength)
        

    
iterate(1,master_segs,'',0)
    
logger.info('bestpath:'+bestpath)
logger.info('record:'+str(record))

logger.info('segments with duplication:')

logger.info('END')




