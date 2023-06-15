# Most code comes from Jimmy Song, Programming Bitcoin, chapter 11

from hash import *
import math

def merkle_parent(hash1, hash2):
    '''Takes the binary hashes and calculates the hash256'''
    # return the hash256 of hash1 + hash2
    return hash256(hash1 + hash2)


# Recall our setting
# We have some hashes in the list hashesOfInteres
# We have a merkle root Mroot
# We want a proof that hashesOfInterest belong to a Merkle tree with the root Mroot

# The proof is of the form proof = (nrLeaves,hashList,flagBits)

# To verify this proof, we construct a MerkleTree filled only with the hashes we can compute

# As a first step we will construct an empty Merkle Tree
# This means a tree that has nrLeaves leaves and the necessary nodes in the levels above
# All the leaves and intermediate nodes have the value 'None'

# Let's code a function that does this

# This is the first task for you: you receive nrLeaves and return a matrix with nodes
# i.e. if nrLeaves = 3, you return:
'''
[ [None],
  [None,None],
  [None,None,None]
]
'''

def EmptyMerkle(nrLeaves):
    
    leaves = nrLeaves # the size of the bottom level

    # compute max depth math.ceil(math.log(self.leaves, 2))
    # This follows since we are just halving at each level
    max_depth = math.ceil(math.log(leaves, 2))
    
    # initialize the nodes property to hold the actual tree
    # each element of the array will be a list containing a level of the tree
    nodes = []
    # loop over the number of levels (max_depth+1)
    for depth in range(max_depth + 1):
        # the number of items at this depth is
        # math.ceil(self.leaves / 2**(self.max_depth - depth))
        # you can work this out; we basically divide by 2 each level
        num_items = math.ceil(leaves / 2**(max_depth - depth))
        # create this level's hashes list with the right number of items
        level_hashes = [None] * num_items
        # append this level's hashes to the merkle tree
        nodes.append(level_hashes)

    return nodes

# Let's create an empty tree

nrLeaves = 15
nodes = EmptyMerkle(nrLeaves)

# Let's see what does this look like

print('Printing an empty tree level by level:')
for level in nodes:
    print(level)
print('\n')



# We can now start coding a PartialMerkleTree class as follows:



# this is our first version; we will extend it afterwards
class PartialMerkleTree1:

    def __init__(self, total):
        self.total = total #nrLeaves
        # compute max depth math.ceil(math.log(self.total, 2))
        self.max_depth = math.ceil(math.log(self.total, 2))
        # initialize the nodes property to hold the actual tree
        self.nodes = []
        # loop over the number of levels (max_depth+1)
        for depth in range(self.max_depth + 1):
            # the number of items at this depth is
            # math.ceil(self.total / 2**(self.max_depth - depth))
            num_items = math.ceil(self.total / 2**(self.max_depth - depth))
            # create this level's hashes list with the right number of items
            level_hashes = [None] * num_items
            # append this level's hashes to the merkle tree
            self.nodes.append(level_hashes)
        # set the pointer to the root (depth=0, index=0)
        self.current_depth = 0
        self.current_index = 0
        # depth tells us the level
        # index tells us which element on this level we are processing
        # this will be used for DFS later on

    # a compact way to print the trees
    def __repr__(self):
        result = []
        for depth, level in enumerate(self.nodes):
            items = []
            for index, h in enumerate(level):
                if h is None:
                    short = 'None'
                else:
                    short = '{}...'.format(h.hex()[:8])
                if depth == self.current_depth and index == self.current_index:
                    items.append('*{}*'.format(short[:-2]))
                else:
                    items.append('{}'.format(short))
            result.append(', '.join(items))
        return '\n'.join(result)


    # we will be filling the tree using DFS
    # do let us define functions we need to traverse the tree

    def up(self):
        # reduce depth by 1 and halve the index
        self.current_depth -= 1
        self.current_index //= 2

    def left(self):
        # increase depth by 1 and double the index
        self.current_depth += 1
        self.current_index *= 2

    def right(self):
        # increase depth by 1 and double the index + 1
        self.current_depth += 1
        self.current_index = self.current_index * 2 + 1

    def root(self):
        return self.nodes[0][0]

    # to allow us define the value of a current node we are processing
    def set_current_node(self, value):
        self.nodes[self.current_depth][self.current_index] = value

    def get_current_node(self):
        return self.nodes[self.current_depth][self.current_index]

    def get_left_node(self):
        return self.nodes[self.current_depth + 1][self.current_index * 2]

    def get_right_node(self):
        return self.nodes[self.current_depth + 1][self.current_index * 2 + 1]

    # as explained in the slides, we will need to check when we reach a leaf
    def is_leaf(self):
        return self.current_depth == self.max_depth

    # if the number of nodes on the next level is odd, the parent hash just needs the left child
    def right_exists(self):
        return len(self.nodes[self.current_depth + 1]) > self.current_index * 2 + 1


# Let us now see how to populate this tree using DFS when we **know** leaf nodes

hex_hashes = [
    "9745f7173ef14ee4155722d1cbf13304339fd00d900b759c6f9d58579b5765fb",
    "5573c8ede34936c29cdfdfe743f7f5fdfbd4f54ba0705259e62f39917065cb9b",
    "82a02ecbb6623b4274dfcab82b336dc017a27136e08521091e443e62582e8f05",
    "507ccae5ed9b340363a0e6d765af148be9cb1c8766ccc922f83e4ae681658308",
    "a7a4aec28e7162e1e9ef33dfa30f0bc0526e6cf4b11a576f6c5de58593898330"
]


tree = PartialMerkleTree1(len(hex_hashes))
# Let's fill in the leaves
# I'll precompute the height manually here
tree.nodes[3] = [bytes.fromhex(h) for h in hex_hashes]

# OK, let's DFS this tree until we can compute the Merkle root:

while tree.root() is None:
    if tree.is_leaf():
        tree.up()
        # why? the leaves are already configured
    else:
        left_hash = tree.get_left_node() # for a non leaf left child always exists
        if left_hash is None: # If we don't have the value we just traverse the child
           tree.left()
        elif tree.right_exists(): 
        # we first verify that the right child exists
        # it might not, if we are in the final node of level i, and level i+1 has an odd number of nodes
            right_hash = tree.get_right_node()
            if right_hash is None: # OK, no value yet, so traverse down
                tree.right()
            else:
                tree.set_current_node(merkle_parent(left_hash,right_hash))
                tree.up()
                # once we have the hashes of both children we can set the value
                # and we move onto the levelabove
        else:
            tree.set_current_node(merkle_parent(left_hash,left_hash))
            tree.up()
            # if there is no right child we are OK with the left one
            # we set the value and move up as before


# Let's see what we have now:

print('Printing a tree filled in by DFS:')
print(tree)
print('\n')

# Here we did a full traversal just to show how we use the tree methods
# Next we explain how to fill in the tree using flagBits
# With flagBits we will be able to compute the Merkle root without filling in the entire tree

# Recall the setting:
#   We have a list of hashes called hashesOfInterest
#   We have a Merkle root Mroot
#   We get proof = (nrLeaves,hashList,flagBits)
#       Using this info we want to compute a Merkle root of a partial tree 
#       And verify it is the same as Mroot
'''
    Algorithm sketch:
    
    1) Check that hashesOfInterest appear in hashList; if not, return false
    2) Create an empty tree = PartialMerkleTree(nrLeaves)

    3) Run a DFS using flagBits and hashList to fill in the blanks:

    Start at the root (as in the exaple above)
    
    Loop until root is computed:

        If processing a leaf:
            it's hash has to be in hashList
            so we pop it from there
            **An set the value of the current node with this hash**
            we also pop the first element of flagBits
            *Traverse* UP one node 

        Else (not a leaf)

            Get left child
            If the hash if not defined:
                pop the first element of flagBits
                    if it's 0 the hash (of the current node) is in hashList
                    so pop it from there and **set the value of the current node**
                    NOTE: this is not the left_child value, but of the current node
                    *Traverse* UP one node  
                else 
                    *traverse* the left subtree (flag bit = 1, so we have to compute it)

            Check that right child exists
                If it does
                Get the right child
                    If the value is not defined:
                        *traverse* the right subtree
                    Else (right_child hash is defined):
                        Compute merkle_parent(left_chiled,right_child)
                        Set this value to the current node
                        *Traverse* UP one node
                Else (right child does not exist):
                    merkle_parent(left_child,left_child)
                    Set this value to the current node
                    *Traverse* UP one node

    When root is computed:
        Raise an error if hashesList is not empty/you reached the last element
        Raise an error if flagBits are not empty/you reached the last element


'''

# We are now ready to populate a partial tree based on this algotihms
# your task is below:

class PartialMerkleTree:

    def __init__(self, total):
        self.total = total #nrLeaves
        # compute max depth math.ceil(math.log(self.total, 2))
        self.max_depth = math.ceil(math.log(self.total, 2))
        # initialize the nodes property to hold the actual tree
        self.nodes = []
        # loop over the number of levels (max_depth+1)
        for depth in range(self.max_depth + 1):
            # the number of items at this depth is
            # math.ceil(self.total / 2**(self.max_depth - depth))
            num_items = math.ceil(self.total / 2**(self.max_depth - depth))
            # create this level's hashes list with the right number of items
            level_hashes = [None] * num_items
            # append this level's hashes to the merkle tree
            self.nodes.append(level_hashes)
        # set the pointer to the root (depth=0, index=0)
        self.current_depth = 0
        self.current_index = 0
        # depth tells us the level
        # index tells us which element on this level we are processing
        # this will be used for DFS later on

    # a compact way to print the trees
    def __repr__(self):
        result = []
        for depth, level in enumerate(self.nodes):
            items = []
            for index, h in enumerate(level):
                if h is None:
                    short = 'None'
                else:
                    short = '{}...'.format(h.hex()[:8])
                if depth == self.current_depth and index == self.current_index:
                    items.append('*{}*'.format(short[:-2]))
                else:
                    items.append('{}'.format(short))
            result.append(', '.join(items))
        return '\n'.join(result)


    # we will be filling the tree using DFS
    # do let us define functions we need to traverse the tree

    def up(self):
        # reduce depth by 1 and halve the index
        self.current_depth -= 1
        self.current_index //= 2

    def left(self):
        # increase depth by 1 and double the index
        self.current_depth += 1
        self.current_index *= 2

    def right(self):
        # increase depth by 1 and double the index + 1
        self.current_depth += 1
        self.current_index = self.current_index * 2 + 1

    def root(self):
        return self.nodes[0][0]

    # to allow us define the value of a current node we are processing
    def set_current_node(self, value):
        self.nodes[self.current_depth][self.current_index] = value

    def get_current_node(self):
        return self.nodes[self.current_depth][self.current_index]

    def get_left_node(self):
        return self.nodes[self.current_depth + 1][self.current_index * 2]

    def get_right_node(self):
        return self.nodes[self.current_depth + 1][self.current_index * 2 + 1]

    # as explained in the slides, we will need to check when we reach a leaf
    def is_leaf(self):
        return self.current_depth == self.max_depth

    # if the number of nodes on the next level is odd, the parent hash just needs the left child
    def right_exists(self):
        return len(self.nodes[self.current_depth + 1]) > self.current_index * 2 + 1




# Your task is to code the populate_tree method
# populate_tree(self, flag_bits, hashes):
# you can assume you already have the number of leaf nodes
# and that you constructed a partial tree
# inside this tree you will populate it according to (flag_bits,hashes)
# at the end I will provide some test examples









    def populate_tree(self, flag_bits, hashes):
        # populate until we have the root
        while self.root() is None:
            # if we are a leaf, we know this position's hash
            if self.is_leaf():
                # get the next bit from flag_bits: flag_bits.pop(0)
                flag_bits.pop(0)
                # set the current node in the merkle tree to the next hash: hashes.pop(0)
                self.set_current_node(hashes.pop(0))
                # go up a level
                self.up()
            else:
                # get the left hash
                left_hash = self.get_left_node()
                # if we don't have the left hash
                if left_hash is None:
                    # if the next flag bit is 0, the next hash is our current node
                    if flag_bits.pop(0) == 0:
                        # set the current node to be the next hash
                        self.set_current_node(hashes.pop(0))
                        # sub-tree doesn't need calculation, go up
                        self.up()
                    else:
                        # go to the left node
                        self.left()
                elif self.right_exists():
                    # get the right hash
                    right_hash = self.get_right_node()
                    # if we don't have the right hash
                    if right_hash is None:
                        # go to the right node
                        self.right()
                    else:
                        # combine the left and right hashes
                        self.set_current_node(merkle_parent(left_hash, right_hash))
                        # we've completed this sub-tree, go up
                        self.up()
                else:
                    # combine the left hash twice
                    self.set_current_node(merkle_parent(left_hash, left_hash))
                    # we've completed this sub-tree, go up
                    self.up()
        if len(hashes) != 0:
            raise RuntimeError('hashes not all consumed {}'.format(len(hashes)))
        for flag_bit in flag_bits:
            if flag_bit != 0:
                raise RuntimeError('flag bits not all consumed')


def verify_inclusion(hashesOfInterest, merkleRoot, nrLeaves, flags,hashes):
    # verifies if hashesOfInterest belong to merkleRoot according to the three-part proofb

    for h in hashesOfInterest:
        if not (h in hashes):
            return False

    leaves = nrLeaves
    flags = flags
    hashes = hashes

    tree = PartialMerkleTree(leaves)
    tree.populate_tree(flags,hashes)

    return (tree.root() == merkleRoot)



###hex_hashes = [
###    "9745f7173ef14ee4155722d1cbf13304339fd00d900b759c6f9d58579b5765fb",
###    "5573c8ede34936c29cdfdfe743f7f5fdfbd4f54ba0705259e62f39917065cb9b",
###    "82a02ecbb6623b4274dfcab82b336dc017a27136e08521091e443e62582e8f05",
###    "507ccae5ed9b340363a0e6d765af148be9cb1c8766ccc922f83e4ae681658308",
###    "a7a4aec28e7162e1e9ef33dfa30f0bc0526e6cf4b11a576f6c5de58593898330"
###]
###
###raw_hashes = [bytes.fromhex(h) for h in hex_hashes]
###
###tree = PartialMerkleTree(5)
###tree.populate_tree([1]*11,raw_hashes)
###
###print(tree)
###
###

hex_hashes = [
    "9745f7173ef14ee4155722d1cbf13304339fd00d900b759c6f9d58579b5765fb",
    "5573c8ede34936c29cdfdfe743f7f5fdfbd4f54ba0705259e62f39917065cb9b",
    "82a02ecbb6623b4274dfcab82b336dc017a27136e08521091e443e62582e8f05",
    "507ccae5ed9b340363a0e6d765af148be9cb1c8766ccc922f83e4ae681658308",
    "a7a4aec28e7162e1e9ef33dfa30f0bc0526e6cf4b11a576f6c5de58593898330",
    "bb6267664bd833fd9fc82582853ab144fece26b7a8a5bf328f8a059445b59add",
    "ea6d7ac1ee77fbacee58fc717b990c4fcccf1b19af43103c090f601677fd8836",
    "457743861de496c429912558a106b810b0507975a49773228aa788df40730d41",
    "7688029288efc9e9a0011c960a6ed9e5466581abf3e3a6c26ee317461add619a",
    "b1ae7f15836cb2286cdd4e2c37bf9bb7da0a2846d06867a429f654b2e7f383c9",
    "9b74f89fa3f93e71ff2c241f32945d877281a6a50a6bf94adac002980aafe5ab",
    "b3a92b5b255019bdaf754875633c2de9fec2ab03e6b8ce669d07cb5b18804638",
    "b5c0b915312b9bdaedd2b86aa2d0f8feffc73a2d37668fd9010179261e25e263",
    "c9d52c5cb1e557b92c84c52e7c4bfbce859408bedffc8a5560fd6e35e10b8800",
    "c555bc5fc3bc096df0a0c9532f07640bfb76bfe4fc1ace214b8b228a1297a4c2",
    "f9dbfafc3af3400954975da24eb325e326960a25b87fffe23eef3e7ed2fb610e",
]

raw_hashes = [bytes.fromhex(h) for h in hex_hashes]

tree = PartialMerkleTree(16)
tree.populate_tree([1]*31,raw_hashes)

print(tree)

flags=[1,0,1,1,0,1,1,0,1,1,0,1,0]

hashes = [
"6382df3f3a0b1323ff73f4da50dc5e318468734d6054111481921d845c020b93",
"3b67006ccf7fe54b6cb3b2d7b9b03fb0b94185e12d086a42eb2f32d29d535918",
"9b74f89fa3f93e71ff2c241f32945d877281a6a50a6bf94adac002980aafe5ab",
"b3a92b5b255019bdaf754875633c2de9fec2ab03e6b8ce669d07cb5b18804638",
"b5c0b915312b9bdaedd2b86aa2d0f8feffc73a2d37668fd9010179261e25e263",
"c9d52c5cb1e557b92c84c52e7c4bfbce859408bedffc8a5560fd6e35e10b8800",
"8636b7a3935a68e49dd19fc224a8318f4ee3c14791b3388f47f9dc3dee2247d1"
]

r_hashes = [bytes.fromhex(h) for h in hashes]

hashesOfInterest = ["9b74f89fa3f93e71ff2c241f32945d877281a6a50a6bf94adac002980aafe5ab",
"c9d52c5cb1e557b92c84c52e7c4bfbce859408bedffc8a5560fd6e35e10b8800"
]

r_interest = [bytes.fromhex(h) for h in hashesOfInterest]


print(verify_inclusion(r_interest, tree.root(), 16, flags,r_hashes))
