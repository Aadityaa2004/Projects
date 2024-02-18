from graph import Graph,Vertex
from priority_queue import PriorityQueue
import sys

def prim(G,S,start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(sys.maxsize)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin()

        if currentVert.getPred() is not None and currentVert.getId() is not None:
            S.addEdge(currentVert.getId(), currentVert.getPred().getId(), currentVert.getDistance())

        for nextVert in currentVert.getConnections():
          newCost = currentVert.getWeight(nextVert)
          if nextVert in pq and newCost<nextVert.getDistance():
              nextVert.setPred(currentVert)
              nextVert.setDistance(newCost)
              pq.decreaseKey(nextVert,newCost)
    return S