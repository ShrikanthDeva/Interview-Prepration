# BFS
```py
# Level wise travel
def bfs(node,vis,ans,graph):
    vis[node] = 1
    q = deque()
    q.append(node)
    while q:
        node = q.popleft()
        ans.append(node)
        for adj in graph[node]:
            if not vis[adj]:
                vis[adj] = 1
                q.append(adj)
    return
```
# DFS
```py
def dfs(node,vis,ans,graph):
    vis[node] = 1
    ans.append(node)
    for adj in graph[node]:
        if not vis[adj]:
            dfs(adj,vis,ans,graph)
    return
```

# Matrix - graph
+ 4 directional:
```py
for dr,dc in [[-1,0],[0,-1],[1,0],[0,1]]:
    nr = r + dr
    nc = c + dc
    if valid(nr,nc):
        # do dfs or bfs
```
+ 8 directional:
```py
for dr,dc in [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[1,1],[1,-1],[-1,1]]:
    nr = r + dr
    nc = c + dc
    if valid(nr,nc):
        # do dfs or bfs
```
---
# DETECT CYCLE
## UNDIRECTED GRAPH

```py
# do dfs carry parent, if node already visited and is not parent then cycle exists
def iscycle(node,parent,vis,graph):
    vis[node] = 1
    for adj in graph[node]:
        if not vis[adj]:
            if iscycle(adj,node,vis,graph):
                return True
        else:
            if adj != parent:
                return True
    return False
```
## DIRECTED GRAPH - DFS
```py
# do dfs mark vis,pathVis -> if already vis and pathVis cycle exists
def iscyle(node,vis,pathVis,graph):
    vis[node] = 1
    pathVis[node] = 1
    for adj in graph[node]:
        if not vis[adj]:
            if iscycle(adj,vis,pathVis,graph):
                return True
        elif pathVis[adj]:
            return True     #visited again in same path ?
    pathVis[node] = 0 #Remove it from the current path
    return False
```
---

# TOPOLOGICAL SORT
## DFS - DAG
```py
# do dfs while returning append to stack
def topo(node,vis,stack,graph):
    vis[node] = 1
    for adj in graph[node]:
        if not vis[adj]:
            topo(adj,vis,stack,graph)
    stack.append(node)
    return stack
```
## TOPOLOGICAL SORT - Kahn's Algo
```py
# detect cycle + topological order
# calculate indegree of all nodes
# Start with nodes having 0 indegree
# Reduce ingree in the order if it becomes 0 append to queue

def topoSort(V, graph):

    indegree = [0 for i in range(V)]
    for i in range(V):
        for j in graph[i]:
            indegree[j] += 1

    q = deque()
    for i in range(V):
        if indegree[i] == 0:
            q.append(i)
    topo = []
    while q:
        node = q.popleft()
        topo.append(node)
        for adj in graph[node]:
            indegree[adj] -= 1
            if indegree[adj] == 0:
                q.append(adj)

    if len(topo) == n:  #cycle does not exists
      return topo
    return False #cycle exists
```
---
# Bipartite Graph -> O( V + 2E )
```py
# Bipartite graph -> nodes can be splitted into 2 groups such that no edges are there btw nodes of the same group but only to another group 
# color the nodes in 0 and 1 , if 2 adj node have same color then not a bipartite
# simple bfs and add opposite color
def isbypartite(node,color,vis,graph):
    vis[node] = 1
    q = deque()
    q.append([node,0])
    color[node] = 0
    while q:
        node,clr = q.popleft()
        for adj in graph[node]:
            if not vis[adj] and color[adj] == -1: # not vis and not colored
                vis[adj] = 1
                color[adj] = clr^1
                q.append([adj,clr^1])
            elif color[adj] == clr: # already visited and colored if same as node's color return false
                return False
    return True
```
---
# Shortest Path
## Undirected Graph -> O( V + 2E )
```py
def bfs(node,vis,dis,graph):
    vis[node] = 1
    dis[node] = 0
    q = deque()
    q.append([node,0])
    while q:
        node,dt = q.popleft()
        for adj in graph[node]:
            if not vis[adj]:
                vis[adj] = 1
                dis[adj] = min(dis[adj],dt+1)
                q.append([adj,dt+1])
    return dis
```
## Directed Graph > Single src SP -> O( V + E )
```py
# Do a topo sort
# traverse the topo stack and compute the distance

def toposort(node,vis,stack,graph):
    vis[node] = 1
    for adj,wt in graph[node]:
        if not vis[adj]:
            toposort(adj,vis,stack,graph)
    stack.append(node)
    
    
graph = dd(list)
for u,v,w in edges:
    graph[u].append([v,w])

# find the topological order as its DAG
vis = [0 for i in range(n)]
s = []
for i in range(n):
    if not vis[i]:
        toposort(i,vis,s,graph)


# traverse the topo and find the dist
dis = [1e9 for i in range(n)]
dis[0] = 0  # given src node
while s:
    node = s.pop()
    for adj,wt in graph[node]:
        dis[adj] = min( dis[adj], dis[node]+wt)
for i in range(n):
    if dis[i] == 1e9:
        dis[i] = -1
return dis    
```
## DIJKSTRA ALGO - Single Src SP -> O( E log(V) )
```py
# Works for both DG and UDG
# Does not work for -ve edge weight or -ve cycles
def dijkstra(self, V, adj, S):

    graph = dd(list)    
    for u in range(V):
        for v,w in adj[u]:
            graph[u].append([v,w])
            graph[v].append([u,w]) #remove if directed
    
    dist = [1e9 for i in range(V)]
    dist[S] = 0

    hq = []
    heapify(hq)
    heappush(hq,[0,S])
    
    while hq:
        dt,node = heappop(hq)
        for nb,wt in graph[node]:
            if dt + wt < dist[nb]:
               dist[nb]  = dt + wt
               heappush(hq,[dist[nb],nb])
    return dist
```

## Print shortest path O( E log(V) + N ) 
```py
def shortestPath(self, n, m, edges):
    graph = dd(list)
    for u,v,w in edges:
        graph[u].append([v,w])
        graph[v].append([u,w])
        
    parent = [i for i in range(n+1)]
    dist = [1e9 for i in range(n+1)]
    dist[1] = 0

    hq = []
    heapify(hq)
    heappush(hq,[0,1]) # [dist,src]

    while hq:
        pathsum,node = heappop(hq)
        for adj,wt in graph[node]:
            if pathsum + wt < dist[adj]:
                dist[adj] = pathsum + wt
                heappush(hq,[dist[adj],adj])
                parent[adj] = node                  # just cache the parent
    
    ans = []
    while(n!=parent[n]):    # trace from dest to src
        ans.append(n)
        n = parent[n]
    if ans:
        ans.append(1)   # add the source node
        return ans[::-1]
    return [-1]
```
# Number of shortest path -> O( E log(V) )
```py
# if u reached a v with minimal cost than previous -> ways[v] = ways[u]
# if u reached a v with same minimal cost like previous -> ways[v] += ways[u]
def countPaths(self, n: int, roads: List[List[int]]) -> int:
    graph = dd(list)
    for u,v,t in roads:
        graph[u].append([v,t])
        graph[v].append([u,t])
        
    ways = [0 for i in range(n)]
    dist = [1e10 for i in range(n)]
    ways[0] = 1
    dist[0] = 0

    q = []
    heapify(q)
    q.append([0,0])
    while q:
        time,node = heappop(q)
        for adj,t in graph[node]:
            if time + t < dist[adj]:
                ways[adj] = ways[node]
                dist[adj] = time + t
                heappush(q,[dist[adj],adj])
            elif time + t == dist[adj]:
                ways[adj] += ways[node]

    return [0,ways[n-1]][dist[n-1]!=1e9]
```
---
# BELLMAN-FORD ALGO - Single Src SP -> O( V.E )
```py
# DG with negative weights or negative cycles
# Single src shortest path
# works only for DG; else convert the UDG to DG by giving both the u->v w and v->u w in edges
# N-1 iterations are enough to compute the distance
# if the array still reduces after N-1th iteration then graph has -ve edges or -ve cycle
def bellman_ford(self, V, edges, S):
        dist = [int(1e8) for i in range(V)]
        dist[S] = 0
        for i in range(V-1):
            for u,v,wt in edges:
                if dist[u] != 1e8 and dist[u]+wt < dist[v]:
                    dist[v] = dist[u] + wt
        # Vth iteration for cycle detection
        for u,v,wt in edges:
            if dist[u] != 1e8 and dist[u]+wt < dist[v]:
                return [-1]
        return dist
```
---
# All Source shortest path

## Floyd - Warshall Algo - All source SP -> O( V.V.V )
```py
# (DG & UDG) with (-ve edge or -ve cycle)
# dist = [[float(inf) for j in range(V)] for i in range(V)]
# dist[i][i] = 0 for all i
# if no direct edge dist[i][j] = inf
# if direct edge dist[i][j] = wt
def floyd(V,matrix):
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j] )
    
    for i in range(V):
        if dist[i][i] < 0:
            return "-VE CYCLE"
    return dist
```
## ALL source SP using dijkstra -> O( V.E.log(V) )
```py
def dijkstra(node,dist,graph):
    dist[node] = 0
    q = []
    heapify(q)
    heappush(q,[0,node])
    while q:
        dt,node = heappop(q)
        for adj,wt in graph[node]:
            if dt+wt < dist[adj]:
                dist[adj] = dt+wt
                heappush(q,[dist[adj],adj])
    

graph = dd(list)
for u,v,wt in edges:
    graph[u].append([v,wt])
    graph[v].append([u,wt])

distance = []
for i in range(n):
    dist = [1e8 for i in range(n)]
    dijkstra(i,dist,graph)
    distance.append(dist)
return distance
```
---
# MST (Minimum Spanning Tree) - DSU - Krushkal's algo
+ MST -> The tree with `n` nodes should only have `n-1` edges and the pathWeight of the tree should be minimal
```py
# sort the edges based on the weights
# traverse -> sorted edges
# take the minimum weight (add the edge & edgeWt) if the nodes does not belong to same component 
#  else skip that edge as it wont contribute to MST

edges = [ [wt,u,v] for u,v,wt in adjlist]
# MlogM
edges.sort()
pathWeight = 0
MST = []
# M.4 alpa. 2
for wt,u,v in edges:
    if find(u) == find(v): #belong to same component
        continue 
    else:
        pathWeight += wt
        unionBySize(u,v)
        MST.append([u,v])
return MST # return pathWeight
```
---
# SCC - Strongly Connected Components - Kosaraju's Algo
```py
# Do a dfs sort the nodes acc to vis time (Topo sort kind of)
# reverse the graph edges
# do the dfs again ; no.of.compo = no.of.times dfs is called
```py
from collections import defaultdict as dd
class Solution:
    
    def dfs(self,node,vis,adj,st):
        vis[node] = 1
        for nb in adj[node]:
            if not vis[nb]:
                self.dfs(nb,vis,adj,st)
        st.append(node)
        
    def dfs2(self,node,vis,adj):
        vis[node] = 1
        for nb in adj[node]:
            if not vis[nb]:
                self.dfs2(nb,vis,adj)

    #Function to find number of strongly connected components in the graph.
    def kosaraju(self, V, adj):
        
        # sorting based on visit time
        st = []
        vis = [0 for i in range(V)]
        for i in range(V):
            if not vis[i]:
                self.dfs(i,vis,adj,st)
                
        # reverse the edges
        graph = dd(set)
        for i in range(V):
            vis[i] = 0 # marking back for the next dfs
            for j in adj[i]:
                graph[j].add(i)
        
        # do the dfs based on the time
        scc = 0
        while st:
            node = st.pop()
            if not vis[node]:
                scc += 1
                self.dfs2(node,vis,graph)
        return scc
```
---
# Tarjan's Algo 
## Detecting BRIDGES in a graph
```py
# A bridge is an EDGE that if removed splits the graph into 2 components
from collections import defaultdict as dd
class Solution:
    timer = 1
    def dfs(self,node,vis,parent,tin,low,graph,bridges):
        vis[node] = 1
        tin[node] = low[node] = self.timer
        self.timer += 1

        for adj in graph[node]:
            # skip the parent
            if adj == parent:
                continue
            # if node is already visited take the minimum
            if vis[adj]:
                low[node] = min(low[node], low[adj])
            else:
                # after dfs comes back check for the adj low time 
                # if the low time is less than the in-time of node -> no problem 
                # as it can reach back in time if this adj-node edge is removed
                # else its an bridge as it cant reach back the node
                self.dfs(adj,vis,node,tin,low,graph,bridges)
                low[node] = min(low[node], low[adj])
                if low[adj] > tin[node]:
                    bridges.append([node,adj])
                    
                    

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:

        graph = dd(set)
        for u,v in connections:
            graph[u].add(v)
            graph[v].add(u)

        vis = [0 for i in range(n)] # check if previously visited or not
        tin = [0 for i in range(n)] # nodes incoming time
        low = [0 for i in range(n)] # lowest time it can reach back
        bridges = []
        self.dfs(0,vis,-1,tin,low,graph,bridges)
        return bridges
```
## Detect ARTICULATION points in a graph
```py
# An articulation point is a NODE on removal splits the graph into multiple components
class Solution:
    timer = 0
    def dfs(self,node,parent,vis,tin,low,mark,adj):
        vis[node] = 1
        tin[node] = low[node] = self.timer
        self.timer += 1
        child = 0
        for nb in adj[node]:
            if nb == parent:
                continue
            # already visited just get the in-time from nb
            if vis[nb]:
                low[node] = min(low[node],tin[nb])
                
            else:
                # after returning from dfs we need to check if we can reach nodes before parent(>=)
                # if not reachable then node is the articulation point
                # here the parent should not be -1 i.e source
                child += 1
                self.dfs(nb,node,vis,tin,low,mark,adj)
                low[node] = min(low[node],low[nb])
                
                if low[nb] >= tin[node] and parent != -1:
                    mark[node] = 1
        # check for the src node
        if child > 1 and parent == -1:
            mark[node] = 1
    #Function to return Breadth First Traversal of given graph.
    def articulationPoints(self, n, adj):
        # code here
        
        vis = [0 for i in range(n)]
        tin = [0 for i in range(n)]
        low = [0 for i in range(n)]
        mark = [0 for i in range(n)]
        
        for i in range(n):
            if not vis[i]:
                self.dfs(i,-1,vis,tin,low,mark,adj)
                
        ans = []
        for i in range(n):
            if mark[i]:
                ans.append(i)
        if ans:
            return ans
        return [-1]
```
---
# DSU -> O(4 alpha)
```py

parent = [i for i in range(V)]
size   = [1 for i in range(V)]
rank   = [0 for i in range(V)]

def find(u):
    if u == parent[u]:
        return u
    parent[u] = find(parent[u])
    return parent[u]

def unionBySize(u,v):
    pu = find(u)
    pv = find(v)

    if pu == pv:    #belong to same component
        return # return True for edge that complete cycle
    if size[pu] > size[pv]:
        pu,pv = pv,pu
    
    parent[pu] = pv
    size[pv] += size[pu]
    return # return False if this edge doesn't complete cycle 

def unionByRank(u,v):
    pu = find(u)
    pv = find(v)

    if pu == pv:    #belong to same component
        return
    if rank[pu] < rank[pv]:
        parent[pu] = pv
    elif rank[pv] < rank[pu]:
        parent[pv] = pu 
    else:   # if equal ranks then increase any of them by 1
        parent[pv] = pu
        rank[pv] += 1
    rank

```
---