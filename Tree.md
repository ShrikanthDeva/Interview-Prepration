# TREE
```py
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```
---
# TRAVERSALS
## In Order
```py
# LEFT - ROOT - RIGHT

#Recursive
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    def inorder(root,ans):
        if not root:
            return 
        inorder(root.left,ans)
        ans.append(root.val)
        inorder(root.right,ans)
        return ans
    return inorder(root,[])


#Iterative
def inorderIterative(self, root: Optional[TreeNode]) -> List[int]:

    inorder = []
    stack = []
    while True:
        if root:
            stack.append(root)
            root = root.left
        elif stack:
            root = stack.pop()
            inorder.append(node.val)
            root = root.right
        else:
            break
    return inorder
```
## Pre Order
```py
# ROOT - LEFT - RIGHT

#Recursive
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    def preorder(root,ans):
        if not root:
            return 
        ans.append(root.val)
        preorder(root.left,ans)            
        preorder(root.right,ans)
        return ans
    return preorder(root,[])

#Iterative
def preorderIterative(self, root: Optional[TreeNode]) -> List[int]:

    preorder = []
    if not root:
        return preorder
    stack = [root]
    while stack:
        root = stack.pop()
        preorder.append(root.val)
        if root.right:
            stack.append(root.right)
        if root.left:
            stack.append(root.left)
    return preorder

```

## Post Order
```py
# LEFT - RIGHT - ROOT

#Recursive
def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    def postorder(root,ans):
        if not root:
            return 
        postorder(root.left,ans)            
        postorder(root.right,ans)
        ans.append(root.val)
        return ans
    return postorder(root,[])

#Iterative
def postorderIterative(self, root: Optional[TreeNode]) -> List[int]:
    
    if not root:
        return []

    stack = [root]
    ans = []
    
    while stack:
        root = stack.pop()
        ans.append(root.val)
        if root.left:
            stack.append(root.left)
        if root.right:
            stack.append(root.right)
    return ans


```
## Level Order
```py
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = []
    ans = []
    q.append([root])
    ans.append([root.val])
    while q:
        x = q.pop(0)
        level_val = []
        level = []
        for i in x:
            if i.left:
                level_val.append(i.left.val)
                level.append(i.left)
                
            if i.right:
                level_val.append(i.right.val)
                level.append(i.right)
        if level:
            ans.append(level_val)
            q.append(level)
    return ans
```
## Zig-Zag Order
```py
def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

    if not root:
        return []
    ans = [[root.val]]
    q = [[root]]
    while q:
        x = q.pop(0)
        level = []
        level_val = []
        for i in x:
            if i.left:
                level.append(i.left)
                level_val.append(i.left.val)
            if i.right:
                level.append(i.right)
                level_val.append(i.right.val)
        if level:
            ans.append(level_val)
            q.append(level)
    
    levelOrder = []
    for i in range(1,len(ans),2):
        levelOrder.append(ans[i][::-1])
    return levelOrder
```

## Boundary Order
```py
def BoundaryOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

    def isLeaf(root):
        if not root.left and not root.right:
            return True
        return False
    
    def addLeftNodes(root,res):
        while root:
            if not isLeaf(root):
                res.append(root.val)
            if root.left:
                root = root.left
            else:
                root = root.right
    
    def addRightNodes(root,res):
        while root:
            if not isLeaf(root):
                res.append(root.val)
            if root.right:
                root = root.right
            else:
                root = root.left
    
    def addLeafNodes(root,res):
        if not root:
            return
        if isLeaf(root):
            res.append(root.val)
        addLeafNodes(root.left,res)
        addLeafNodes(root.right,res)
    

    ans = []
    if not root:
        return ans
    ans.append(root.val)
    addLeftNodes(root.left,ans)
    addLeafNodes(root,ans)
    addRightNodes(root.right,ans)

    return ans
```

## Vertical Order
```py
from collections import defaultdict as dd
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:

    if not root:
        return []
    ans = dd(list)
    ans[0].append([0,root.val])
    # xaxis,height,node
    q = [ [[0,0,root]]]
    while(q):
        clvl = q.pop(0)
        nlvl = []
        for x in clvl:
            xa,ht,node = x
            if node.left:
                nlvl.append([xa-1,ht+1,node.left])
                ans[xa-1].append([ht+1,node.left.val])
            if node.right:
                nlvl.append([xa+1,ht+1,node.right])
                ans[xa+1].append([ht+1,node.right.val])
        if nlvl:
            q.append(nlvl)
    op = []
    # sorting based on x-axis
    for k,val in sorted(ans.items()):
        x = []
        # sorting based on height
        for i in sorted(val):
            x.append(i[1])
        op.append(x)
    return op
```

## Top View

+ Nodes with the lowest height in each x-axis chosen

```py
def getTopView(root):

    ds = dd(list)

    def inorder(root,val,ht,ds):
        if not root:
            return
        inorder(root.left,val-1,ht+1,ds)
        if val not in ds:
            ds[val] = [ht,root.val]
        else:
            if ht < ds[val][0]:
                ds[val] = [ht,root.val]
        inorder(root.right,val+1,ht+1,ds)
    
    inorder(root,0,0,ds)
    # print(dict(ds))
    op = []
    for k,v in sorted(ds.items()):
        op.append(v[1])
    return op
```

## Bottom View

+ Nodes with highest hight in each x-axis

```py
def bottomView(root):

    d = dd(list)

    def inorder(root,xaxis,ht,d):
        if not root:
            return
        inorder(root.left,xaxis-1,ht+1,d)
        if xaxis not in d:
            d[xaxis] = [ht,root.data]
        else:
            if d[xaxis][0] < ht:
                d[xaxis] = [ht,root.data]
        inorder(root.right,xaxis+1,ht+1,d)
    inorder(root,0,0,d)
    ans = []
    for k,v in sorted(list(d.items())):
        ans.append(v[1]) 
    return ans
```

## Right View

+ Nodes with highest x-axis in each height

```py
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:

    d = dd(list)

    def inorder(root,xaxis,ht,d):
        if not root:
            return
        inorder(root.left,xaxis-1,ht+1,d)
        d[ht] = [xaxis,root.val]
        inorder(root.right,xaxis+1,ht+1,d)

    inorder(root,0,0,d)
    ans = [d[k][1] for k in sorted(d)]
    return ans 
```
---
# Other Important Stuffs

## Height of a BT
```py
def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max( 1+ self.maxDepth(root.left), 1+ self.maxDepth(root.right))
```
## Balanced BT
```py
def isBalanced(self, root: Optional[TreeNode]) -> bool:
    def check(root):
        if not root:
            return True
        
        lh = check(root.left)
        rh = check(root.right)
        
        if lh == -1 or rh == -1 or abs(lh-rh) > 1:
            return -1
        return 1+max(lh,rh)
    return check(root) != -1
```
## Diameter of a BT
```py
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

    # lh + rh 
    self.diameter = 0

    def findHeight(root):
        if not root:
            return 0
        
        lh = findHeight(root.left)
        rh = findHeight(root.right)

        self.diameter = max(self.diameter,lh+rh)

        return 1+max(lh,rh)
    findHeight(root)
    return self.diameter
```
## Max-Path-Sum
```py
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    self.maxsum = -float(inf)
    def maxps(root):
        if not root:
            return 0
        # not considering the negative values
        mpsl = max(0,maxps(root.left))
        mpsr = max(0,maxps(root.right))
        # updating the max path sum
        self.maxsum = max( self.maxsum, root.val + mpsl + mpsr )
        #  you can only choose one direction so returning the maxpathsum direction
        return root.val + max( mpsl, mpsr)
    maxps(root)
    return self.maxsum
```
## Check-Identical
```py
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p and q:
            if p.val == q.val:
                return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
            return False
        if not p and not q:
            return True
        else:
            return False
```
## Check-Symmetrical
+ Check whether it is a mirror of itself (i.e., symmetric around its center).
```py
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        
    def isSymm(r1,r2):
        if r1 and r2:
            if r1.val == r2.val:
                return isSymm(r1.left,r2.right) and isSymm(r1.right,r2.left)
            else:
                return False
        elif not r1 and not r2:
            return True
        else:
            return False
            
    return isSymm(root.left,root.right)
```

## Lowest-Common-Ancestor in BT
```py
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
    def postorder(root,p,q):
        if not root:
            return None
        if root.val == p.val:
            return p
        if root.val == q.val:
            return q
        
        la = postorder(root.left,p,q)
        ra = postorder(root.right,p,q)

        if la and ra:
            return root
        elif la:
            return la
        elif ra:
            return ra
        else:
            return None
    return postorder(root,p,q)
```

## Max-Width of BT
```py
def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:

    self.maxWidth = 1
    q = [[root,1]]
    while q:
        n = len(q)
        for i in range(n):
            node,idx = q.pop(0)
            if node.left:
                q.append([node.left,2*idx-1])
            if node.right:
                q.append([node.right,2*idx])
        if q:   
            self.maxWidth = max(self.maxWidth, q[-1][1]-q[0][1] + 1 )
    return self.maxWidth
```
## Time to burn a Tree From a given node
```py
def timeToBurnTree(root, start):

    d = dd(set)

    q = [root]
    while q:
        
        node = q.pop(0)
        if node.left:
            d[node.data].add(node.left.data)
            d[node.left.data].add(node.data)
            q.append(node.left)
        if node.right:
            d[node.data].add(node.right.data)
            d[node.right.data].add(node.data)
            q.append(node.right)

    t = 0
    q = [start]
    vis = set()
    vis.add(start)
    while q:
        n = len(q)
        for i in range(n):
            node = q.pop(0)
            for nb in d[node]:
                if nb not in vis:
                    q.append(nb)
                    vis.add(nb)
        t += 1
    return t-1
```

## Count Nodes
```py
c = 0
def countNodes(self, root: Optional[TreeNode]) -> int:
    def count(root):
        if not root:
            return
        self.c+=1
        count(root.left)
        count(root.right)
    count(root)
    return self.c
```

## Flatten BT
```py
def flatten(self, root: Optional[TreeNode]) -> None:
        
    self.prev = None
    def flat(root):
        if not root:
            return None
        flat(root.right)
        flat(root.left)
        root.left = None
        root.right = self.prev
        self.prev = root
        return
    flat(root)
    return root
```
## Construct Binary Tree from Preorder and Inorder Traversal
```py
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        # if no inorder return null -> no child
        if not inorder :
            return None

        # first node in preorder is the root
        x = TreeNode( preorder[0] )

        # finding the location of root in inorder -> to split the left subtree and right subtree
        loc = 0
        while( loc < len(inorder) ):
            if inorder[loc] == preorder[0]:
                break
            loc += 1

        # left subtree -> p[ from next ele : no.of left-children+1 ] , i[ start : no.of left-children]
        x.left = self.buildTree( preorder[1: loc+1 ], inorder[ : loc] )

        # right subtree -> p[ from root loc next ele : end ] , i[from root loc next ele : end]
        x.right = self.buildTree( preorder[ loc+1: ], inorder[ loc+1: ] )

        # return subtree-> which gets added to its parent recursively
        return x
```

## Construct Binary Tree from Inorder and Postorder Traversal

```py
def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        d = dd(int)
        for i in range(len(inorder)):
            d[inorder[i]] = i
        
        def BT(inorder,postorder,istart,iend,pstart,pend):
            if istart > iend or pstart > pend:
                return None
            x = TreeNode(postorder[pend])
            inroot = d[x.val]
            rlen = iend-inroot

            x.left  = BT(inorder,postorder,istart,inroot-1,pstart,pend-rlen-1)
            x.right = BT(inorder,postorder,inroot+1,iend,pend-rlen-1,pend-1)

            return x
        return BT(inorder,postorder,0,len(inorder)-1,0,len(postorder)-1)
```
---

# Binary Search Trees

## Insert node Into BST
```py
def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def bs(root,par,val):
            if not root:
                return par
            if val > root.val:
                return bs(root.right,root,val)
            else:
                return bs(root.left,root,val)
        if not root:
            return TreeNode(val)
        node = bs(root,root,val)
        if node.val > val:
            node.left = TreeNode(val)
        else:
            node.right = TreeNode(val)
        return root
```

## Delete Node Into BST

```py
def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:

        def bs(root,par,val):
            if not root:
                return -1,-1
            if root.val > val:
                return bs(root.left,root,val)
            elif root.val == val:
                return root,par
            else:
                return bs(root.right,root,val)
        node,par = bs(root,root,key)
        if node == -1:
            return root
        if node.val < par.val:
            a,b = node.left,node.right
            if a and b:
                leftmost = b
                while(leftmost and leftmost.left):
                    leftmost = leftmost.left
                leftmost.left = a.right
                a.right = b
                par.left = a
            elif a:
                par.left = a
            elif b:
                par.left = b
            else:
                par.left = None
        elif node.val > par.val:
            a,b = node.left,node.right
            if a and b:
                rightmost = a
                while(rightmost and rightmost.right):
                    rightmost = rightmost.right
                rightmost.right = b.left
                b.left = a
                par.right = b
            elif a:
                par.right = a
            elif b:
                par.right = b
            else:
                par.right = None
        else:
            if par.left and par.right:
                a,b = par.left ,par.right
                while(b and b.left):
                    b = b.left
                b.left = a.right
                a.right = par.right
                root = a
            elif par.left:
                return par.left
            elif par.right:
                return par.right
            else:
                return None

        return root
```

## K-th Smallest Element in BST

```py
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    self.count = 0
    def inorder(root,k):
        if not root:
            return 
        inorder(root.left,k)
        self.count += 1
        if self.count == k:
            self.ans = root.val
        inorder(root.right,k)
    inorder(root,k)
    return self.ans
```
## Validate BST
```py
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def validate(root,low = -float(inf),high = float(inf)):
        if not root:
            return True
        if root.val <= low or root.val >= high:
            return False
        return validate(root.right,root.val,high) and validate(root.left,low,root.val)
    return validate(root)
        
```
## LCA in BST
```py
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    if p.val < root.val and q.val < root.val:
        return self.lowestCommonAncestor(root.left,p,q)
    elif p.val > root.val and q.val > root.val:
        return self.lowestCommonAncestor(root.right,p,q)
    else:
        return root
```
## Construct BST from Preorder
```py
def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        self.cur = 0
        def createbst(preorder,ub=float(inf)):
            if self.cur >= len(preorder) or preorder[self.cur] >= ub:
                return None
            root = TreeNode(preorder[self.cur])
            self.cur += 1
            root.left = createbst(preorder,root.val)
            root.right = createbst(preorder,ub)
            return root
        return createbst(preorder,float(inf))
```

## 2-Sum in BST
```py
def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        d = dd(int)
        def hastarget(root,k,d):
            if not root:
                return False
            if k-root.val in d:
                return True
            d[root.val] = 1
            return hastarget(root.left,k,d) or hastarget(root.right,k,d)
        return hastarget(root,k,d)
```
## Recover BST
```py
def recoverTree(self, root: Optional[TreeNode]) -> None:
    self.prev = None
    def recover(root):
        if not root:
            return 
        recover(root.left)
        # first violation
        if self.prev!= None and root.val < self.prev.val:
            if not self.first:
                self.first = self.prev
                self.second = root
            else:
                self.third = root
        self.prev = root
        recover(root.right)
```
