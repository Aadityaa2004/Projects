"""
UMass ECE 241 - Advanced Programming
Project 2 - Fall 2023
"""
import sys
from graph import Graph, Vertex
from priority_queue import PriorityQueue
from prims import prim
from dikstras import dijkstra

class DeliveryService:
    def __init__(self) -> None:
        """
        Constructor of the Delivery Service class
        """
        self.city_map = Graph() # Initialize the city map as an instance of the Graph class
        self.MST = Graph() # Initialize the Minimum Spanning Tree (MST) as an instance of the Graph class

    def buildMap(self, filename: str) -> None:
        '''
        This function is used to open the map txt file. It extracts information from the
        file in such a way we have the start_node, end_node and cost. We further create a 
        graph function and add the edges to the list. 
        '''


        file = open(filename, 'r') # opens the file only under the option of "read only"
        for line in file:
            extract_file = line.strip().split('|') # extract all the data from the file
            if extract_file:
                start_node = int(extract_file[0])
                end_node = int(extract_file[1])
                cost = int(extract_file[2])

                # Add the edge to the city_map graph
                self.city_map.addEdge(start_node,end_node,cost)     

    def isWithinServiceRange(self, restaurant: int, user: int, threshold: int) -> bool:
        """
        We apply the bellmaford algorithm under this function to find the shortest path. 
        It returns True if the user is within the service range, and False otherwise. 
        """
        if restaurant not in self.city_map.getVertices() or user not in self.city_map.getVertices():
            return False
        
        dist = {}
        # Step 1: Initialize distances from src to all other vertices as INFINITE
        for v in self.city_map.getVertices():
            if v  == restaurant:
                dist[v] = 0
            else:
                dist[v] = float('inf')

        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edge
        for _ in range(len(self.city_map.getVertices())-1):
        
            for v in self.city_map:
                # Update dist value and parent index of the adjacent vertices of
                # the picked vertex. Consider only those vertices which are still in
                # queue
                for nextVert in v.getConnections():
                    if dist[v.id] != float("Inf") and int(dist[v.id]) + v.getWeight(nextVert) < dist[nextVert.id]:
                        dist[nextVert.id] = dist[v.id] + v.getWeight(nextVert)

        if dist[user] <= threshold:
            return True
        else: 
            return False

    def buildMST(self, restaurant: int) -> bool:
        """
        Builds a Minimum Spanning Tree (MST) starting from a specified restaurant. It uses the prim
        function to construct the MST. Returns True if the restaurant is a valid vertex, and False 
        otherwise.
        """
        if restaurant not in self.city_map.getVertices():
            return False
        else:
            # We have created a prim.py file where we import the prims algorithm to 
            # create a minimum spanning tree.
            return prim(self.city_map,self.MST,self.city_map.getVertex(restaurant))

    def minimalDeliveryTime(self, restaurant: int, user: int) -> int:
        """
        Calculates the minimal delivery time from a restaurant to a user within the MST.
        Uses Dijkstra's algorithm on the MST.
        Returns the minimal delivery time as an integer.
        Returns -1 if either the user or restaurant is not in the MST.
        """
        if user is not None and user in self.MST.getVertices() and restaurant is not None and restaurant in self.MST.getVertices():
            return dijkstra(self.MST,self.MST.getVertex(restaurant),self.MST.getVertex(user),delay=None,B=True)
            # in the above part, we call the dijkstra's algoritm from the dijkstra.py
        else:
            return -1

    def findDeliveryPath(self, restaurant: int, user: int) -> str:
        """
        Finds the delivery path from a restaurant to a user without considering delays.
        Uses Dijkstra's algorithm on the original city map.
        Returns the delivery path as a string.
        Returns "INVALID" if either the user or restaurant is not in the city map.
        """
        if user not in self.city_map.getVertices() and restaurant not in self.city_map.getVertices():
            return "INVALID"
        else:
            # in the above part, we call the dijkstra's algoritm from the dijkstra.py
            return dijkstra(self.city_map,self.city_map.getVertex(restaurant),self.city_map.getVertex(user),delay=None,B=False)

    def findDeliveryPathWithDelay(self, restaurant: int, user: int, delay_info: dict[int, int]) -> str:
        """
        Finds the delivery path from a restaurant to a user considering delays.
        Uses Dijkstra's algorithm on the original city map with delay information.
        Returns the delivery path as a string.
        Returns "INVALID" if either the user or restaurant is not in the city map.
        """
        
        if user not in self.city_map.getVertices() and restaurant not in self.city_map.getVertices():
            return "INVALID"
        # in the above part, we call the dijkstra's algoritm from the dijkstra.py
        return dijkstra(self.city_map,self.city_map.getVertex(restaurant),self.city_map.getVertex(user),delay=delay_info,B=False)



    ## DO NOT MODIFY CODE BELOW!
    @staticmethod
    def nodeEdgeWeight(v):
        return sum([w for w in v.connectedTo.values()])

    @staticmethod
    def totalEdgeWeight(g):
        return sum([DeliveryService.nodeEdgeWeight(v) for v in g]) // 2

    @staticmethod
    def checkMST(g):
        for v in g:
            v.color = 'white'

        for v in g:
            if v.color == 'white' and not DeliveryService.DFS(g, v):
                return 'Your MST contains circles'
        return 'MST'

    @staticmethod
    def DFS(g, v):
        v.color = 'gray'
        for nextVertex in v.getConnections():
            if nextVertex.color == 'white':
                if not DeliveryService.DFS(g, nextVertex):
                    return False
            elif nextVertex.color == 'black':
                return False
        v.color = 'black'

        return True

# NO MORE TESTING CODE BELOW!
# TO TEST YOUR CODE, MODIFY test_delivery_service.py
