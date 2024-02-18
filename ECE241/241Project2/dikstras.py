from graph import Graph,Vertex
from priority_queue import PriorityQueue

def dijkstra(aGraph,start,end,delay,B):
    if start is None:
        return 'INVALID'
    
    for v in aGraph:
        v.setDistance(float('Inf'))
        v.setPred(None)

    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() \
                    + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance( newDist )
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert,newDist)
    
    if B is True:
        return end.getDistance()
    else:
        if delay == None:
            if start == None:
                return 'INVALID'
            if end == None:
                return "INVALID"
            path = [] 
            curr = end
            while curr:
                path.insert(0,str(curr.getId()))
                curr = curr.getPred()
            path_str = "->".join(path)
            return f"{path_str} ({end.getDistance()})"
        else:

            path = [] 
            curr = end
            inc_time = 0
            while curr:
                if curr.getId() in delay:
                    inc_time += delay[curr.getId()]
                path.insert(0,str(curr.getId()))
                curr = curr.getPred()
            path_str = "->".join(path)
            return f"{path_str} ({end.getDistance() + inc_time})"
            