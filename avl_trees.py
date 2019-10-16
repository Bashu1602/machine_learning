import numpy as np
from heapq import heappush, heappop
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

class Node():
    def __init__(self, proj, point, parent = None):
        self.proj = proj
        self.points = []
        self.points.append(point)
        self.parent = parent
        self.left = None
        self.right = None
        self.height = 0
        
    def has_left(self):
        return True if self.left != None else False
    
    def has_right(self):
        return True if self.right != None else False
    
    def is_left(self):
        return True if self == self.parent.left else False
    
    def is_right(self):
        return True if self == self.parent.right else False
    
    def is_root(self):
        return True if self.parent == None else False

    
class AVLTree():
    def __init__(self):
        self.root = None
        
    def insert(self, proj, point, currentNode=None):
        '''
        Inserts a node into the subtree rooted at this node.
 
        Args:
            currentNode: The node to be inserted.
            proj: Projection of a point onto unit vector
            point: Query Point
        Time Complexity: O(log(n))
        '''
        if self.root == None:
            self.root = Node(proj, point)
            return
            
        if proj < currentNode.proj:
            if currentNode.left is None:
                currentNode.left = Node(proj, point, parent = currentNode)
                self.update_balance(currentNode.left)
            else:self.insert(proj, point, currentNode.left)
                
                
        elif proj > currentNode.proj:
            if currentNode.right is None:
                currentNode.right = Node(proj, point, parent = currentNode)
                self.update_balance(currentNode.right)
                
            else:self.insert(proj, point, currentNode.right)
     
        else:
            currentNode.points.append(point)
                
    def update_balance(self, node):
        '''
        Time Complexity: O(log(n))
        '''
        if node.height < -1 or node.height > 1:
            self.rebalance(node)
            return
        if node.parent != None:
            if node.is_left():
                node.parent.height += 1
            elif node.is_right():
                node.parent.height -= 1
            if node.parent.height != 0:
                self.update_balance(node.parent)
    
    def rebalance(self, node):
        '''
        Time Complexity: O(C)
        If tree is out of balance (it's left and right subtrees height differ by more than abs(1)), than we need to rebalance it.
        Balancing is done by single left or right rotations or with double left or right rotations of the tree.
        '''
        if node.height < 0:
            if node.right.height > 0:
                self.right_rotate(node.right)
                self.left_rotate(node)
            else:
                self.left_rotate(node)
        elif node.height > 0:
            if node.left.height < 0:
                self.left_rotate(node.left)
                self.right_rotate(node)
            else:
                self.right_rotate(node)
    '''
    Rotation
    Tree can be rotated left or right.
    With left rotation, right subtree root replaces current root. With right rotation, left subtree replaces current root.
    '''
    def left_rotate(self, old_root):
        '''
        Time Complexity: O(C)
        '''
        new_root = old_root.right
        
        old_root.right = new_root.left
        if new_root.left != None:
            new_root.left.parent = old_root
        
        new_root.parent = old_root.parent
        
        if old_root.is_root():
            self.root = new_root
        else:
            if old_root.is_left():
                old_root.parent.left = new_root
            else:
                old_root.parent.right = new_root
        
        new_root.left = old_root
        old_root.parent = new_root
        
        old_root.height = old_root.height + 1 - min(new_root.height, 0)
        new_root.height = new_root.height + 1 + max(old_root.height, 0)
        
    def right_rotate(self, old_root):
        '''
        Time Complexity: O(C)
        '''
        new_root = old_root.left
        
        old_root.left = new_root.right
        if new_root.right != None:
            new_root.right.parent = old_root

        new_root.parent = old_root.parent
        
        if old_root.is_root():
            self.root = new_root
        else:
            if old_root.is_left():
                old_root.parent.left = new_root
            else:
                old_root.parent.right = new_root
                
        new_root.right = old_root
        old_root.parent = new_root
        
        old_root.height = old_root.height - 1 - max(new_root.height, 0)
        new_root.height = new_root.height - 1 - min(old_root.height, 0)
        
    def predecessor(self, root, pred, proj):
        '''
        Time Complexity: O(log(n))
        '''
        if root is None:
            return None
        if proj < root.proj:
            if root.has_left():
                return self.predecessor(root.left, pred, proj)
            else:
                return pred
        elif proj == root.proj:
            if root.has_left():
                pred = self.maximum(root.left)
            return pred
        else:
            if root.has_right():
                pred = root
                return self.predecessor(root.right, pred, proj)
            else:
                return root
                
    def maximum(self, root):
        '''
        Time Complexity: O(log(n))
        '''
        while root.has_right():
            root = root.right
        return root
    
    def successor(self, root, succ, proj):
        '''
        Returns the node that contains the successor proj 
        
        Args:
            proj: Projection onto the unit vector
 
        Returns:
            The successor node.
            
        Time Complexity: O(log(n))
        '''
        if root is None:
            return None
        if proj < root.proj:
            if root.has_left():
                succ = root
                return self.successor(root.left, succ, proj)
            else:
                return root
        elif proj == root.proj:
            if root.has_right():
                succ = self.minimum(root.right)
            return succ
        else:
            if root.has_right():
                return self.successor(root.right, succ, proj)
            else:
                return succ
            
    def minimum(self, root):
        '''
        Time Complexity: O(log(n))
        '''
        while root.has_left():
            root = root.left
        return root
    
    def closer(self, pred, succ, proj):
        '''
        Time Complexity: O(C)
        '''
        if abs(pred.proj-proj) < abs(succ.proj-proj):
            return [pred, succ]
        elif abs(pred.proj-proj) > abs(succ.proj-proj):
            return [succ, pred]
        else:
            return [pred, succ]
        
    def search(self, current_node, proj):
        '''
        Time Complexity: O(log(n))
        '''
        if self.root is None:
            return None
        if proj < current_node.proj:
            if current_node.has_left():
                return self.search(current_node.left, proj)
            else:
                return None
        elif proj > current_node.proj:
            if current_node.has_right():
                return self.search(current_node.right, proj)
            else:
                return None
        else:
            return current_node
    
    def sort_tree(self, proj, n):
        '''
        Given a projection of a point, sort avl tree, return a list of nodes.
        closest = [node1, node2, ..., node3]
        '''
        if self.root is None:
            return None
        closest = []
        node = self.search(self.root, proj)
        pred = self.predecessor(self.root, None, proj)
        succ = self.successor(self.root, None, proj)
        if node is not None:
            closest += [node]
        while len(closest) < n:
            if pred is not None and succ is not None:
                closest += self.closer(pred, succ, proj)
            elif pred is not None and succ is None:
                closest += [pred]
            elif pred is None and succ is not None:
                closest += [succ]
            else:
                pass
            if pred is not None:
                pred = self.predecessor(self.root, None, pred.proj)
            if succ is not None:
                succ = self.successor(self.root, None, succ.proj)
        return closest

def query(sorted_tree, k):
    '''
    Return kth closest node
    '''
    return sorted_tree[k].proj, sorted_tree[k].points[0]

    
def CONSTRUCT(D, simple_ind, comp_ind, q):
    '''
    Time complexity: O(mL(nd+nlogn))
    Construct unit_vec, projs and trees
    '''
    dims = D.shape[1]
    unit_vec = np.zeros((simple_ind,comp_ind),object)
    trees = np.zeros((simple_ind,comp_ind),object)
    
    for j in range(simple_ind):
        for l in range(comp_ind):
            v = np.random.normal(0,1,dims)
            uvec = v / np.dot(v,v)**0.5
            unit_vec[j,l] = uvec
            projs = np.dot(D, uvec)
            trees[j,l] = AVLTree()
            for i in range(len(projs)):
                trees[j,l].insert(projs[i],i,trees[j,l].root)

    sorted_trees = np.zeros((simple_ind,comp_ind),object)
    query_proj = np.zeros((simple_ind,comp_ind))
    for j in range(simple_ind):
        for l in range(comp_ind):
            query_proj[j,l] = np.dot(unit_vec[j,l], q)
            sorted_trees[j,l] = trees[j,l].sort_tree(query_proj[j,l], n)
            

    return unit_vec, trees, sorted_trees, query_proj

def euclidean_dist(p, q):
    '''
    Time Complexity: O(d)
    '''
    return np.dot(p-q,p-q)**0.5

def QUERY(q, unit_vec, trees, sorted_trees, query_proj, D, k0, k1, k):
    '''
    Time Complexity:
    heapq.push() and heapq.pop(): O(log(m))
    k1 refers how many points popped from a heap
    k0 refers how many candidates a composite index can have
    '''
    n = D.shape[0]
    m = unit_vec.shape[0]
    L = unit_vec.shape[1]
    Cls = np.zeros((L, n))
    Sls = np.zeros(L, dtype=object)
    for l in range(L):
        Sls[l] = set()
    Pls = np.zeros(L, dtype=object)  
    for l in range(L):
        Pls[l] = []


    for l in range(L):
        for j in range(m):
            p_proj, pt = query(sorted_trees[j,l], 0)
            priority = abs(p_proj - query_proj[j,l])
            heappush(Pls[l], (priority, pt, j, l, 0))
    
    count = 0
    for i in range(k1):
        for l in range(L):
            if len(Sls[l]) < k0:
                cp_Pl = Pls[l][:]
                while(cp_Pl[0][4] == n-1):
                    heappop(cp_Pl)
                popped_pt = None
                if len(cp_Pl) == 0:
                    popped_pt = heappop(Pls[l])[1]
                else:
                    popped_pt = heappop(Pls[l])[1]
                    point, origin_j, origin_l, ith = cp_Pl[0][1:5]
                    p_proj, pt = query(sorted_trees[origin_j, origin_l], ith+1)
                    priority = abs(p_proj - query_proj[origin_j, origin_l])
                    heappush(Pls[l], (priority, pt, origin_j, origin_l, ith+1))
                Cls[l, popped_pt] += 1
                if Cls[l, popped_pt] == 70 * m / 100:
                    Sls[l].add(popped_pt)
                count += 1

    
    print ("The number of points visisted:", count)
    for l in range(L):
        print ("The number of points in "+ str(l) +"th candidate set:",len(Sls[l]))
    candidates = set()
    for l in range(L):
        candidates = candidates.union(Sls[l])
    print ("The number of candidates:",len(candidates)) 
    candi_pt = []
    candi_eudist = []
    for pt in candidates:
        candi_pt.append(pt)
        candi_eudist.append(euclidean_dist(D[pt], q))
    
    k_neighbours = []
    sorted_eudist = np.argsort(candi_eudist)
    i = 0
    while i < k:
        k_neighbours.append(candi_pt[sorted_eudist[i]])
        i += 1
    
    return np.array(k_neighbours)

def bruteforce(q,dataset,k):
    '''
    Time complexity: O(dn + nlog(n))
    '''
    dataset_eu_dist = []
    for i in range(len(dataset)):
        dataset_eu_dist.append(euclidean_dist(q,dataset[i]))
    return np.argsort(dataset_eu_dist)[:k]

def accuracy(pre, gold):
    count = 0
    for pt in pre:
        if pt in gold:
            count += 1
    return float(count) / len(gold)

def scatter_plot(D, Y):
    plt.scatter(D, Y, c="blue", alpha=0.5)
    plt.scatter(10, -5, c="Red", alpha=0.5)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    return plt.show()

if __name__ == "__main__":
    n = 1000
    ambient_dim = 20
    intrinsic_dim = 20
    simple_ind = 20
    comp_ind = 20
    D, labels_true = make_blobs(n_samples=n, n_features=ambient_dim, centers=10, cluster_std=5,random_state=0)    
    #mnist = fetch_mldata('MNIST original')
    #D = mnist.data
    #print len(D[1])
    #For understanding ploting of data points and query point
    Y, labels_true = make_blobs(n_samples=n, n_features=ambient_dim, centers=10, cluster_std=5,random_state=5)
    print("Scatter Plot", scatter_plot(D, Y))
    k = 10
    pt = 50
    q = D[pt]
    k0 = int(k * max(np.log(n/k), (n/k)**(1-float(simple_ind)/intrinsic_dim)))
    k1 = int(simple_ind * k * max(np.log(n/k), (n/k)**(1-float(1)/intrinsic_dim)))

    print ("*************** To find top",k,"nearest points from",n,"points ***************")
    print ("Configuration:")
    print ("Ambient dimensionality \t\t\t\t\t\tambient_dim =",ambient_dim)
    print ("Intrinsic dimensionality \t\t\t\t\tintrinsic_dim =",intrinsic_dim)
    print ("The number of simple indices \t\t\t\t\tsimple_ind =",simple_ind)
    print ("The number of composite indices \t\t\t\tcomp_ind =",comp_ind)
    print ("The number of points to retrieve for one composite index \tk0 =",k0)
    print ("The number of points to visit for one composite index \t\tk1 =",k1)
    print ("the query point \t\t\t\t\t\t\t\tquery_point = ",pt)
    print ("******************************************************************************\n")

    print ("******************************************************************************")
    print ("Construction\n")
    unit_vec, trees, sorted_trees, query_proj= CONSTRUCT(D, simple_ind, comp_ind, q)
    print ("Done!")
    print ("******************************************************************************\n")

    print ("******************************************************************************")
    print ("Algorithm 1 - Brute Force\n")
    bf_points = bruteforce(q, D, k)
    print ("Output:",list(bf_points))
    print ("******************************************************************************\n")
    
    print ("******************************************************************************")
    print ("Algorithm 2 - PDCI\n")
    print ("k0=",k0)
    print ("k1=",k1)
    pdci_points = QUERY(q, unit_vec, trees, sorted_trees, query_proj, D, k0, k1, k)
    print ("Output:\n",list(pdci_points))
    print ("Accuracy:",accuracy(pdci_points, bf_points))
    print ("******************************************************************************\n")