from hash import hash
import json

class Block:
	def __init__(self, data, prev_hash=None):
		self.data = data
		self.prev_hash = prev_hash

	def serialize(self):
		if self.prev_hash == None:
			d = { 'data': self.data }
		else:
			d = { 'data': self.data,
			      'prev_hash': self.prev_hash }
		return bytes(json.dumps(d, sort_keys=True).encode('utf-8'))

	def __str__(self):
		return '(Data: {} - Previous: {})'.format(self.data, self.prev_hash)

class Blockchain:
	def __init__(self):
		self.elements = {}

	def add_block(self, block):
		
		if ( (block.prev_hash in self.elements.keys()) or 
			 (len(self.elements.keys()) == 0) and (block.prev_hash == None)):

			serialize_to_hash = hash(block.serialize()).hex()
			self.elements[serialize_to_hash] = block
			return serialize_to_hash

		else:
			return None


	def check(self, block_hash):
		if (block_hash not in self.elements.keys()):
			return False

		serialize_to_hash = hash(self.elements[block_hash].serialize()).hex()
		if (block_hash != serialize_to_hash):
			return False

		current = self.elements[block_hash]
		while (current.prev_hash):
			if (current.prev_hash not in self.elements.keys()):
				return False

			serialize_to_hash = hash(self.elements[current.prev_hash].serialize()).hex()
			if (current.prev_hash != serialize_to_hash):
				return False

			current = self.elements[current.prev_hash]

		return True

	def print_blockchain(self):
		for key in self.elements.keys():
			print('Key: {}'.format(key))
			print('Block: {}'.format(self.elements[key]))


#############
##EXAMPLE 1##
#############

blockchain = Blockchain()

genesis = Block('1', None)
prev = blockchain.add_block(genesis)
bl1 = Block('2', prev)
prev2 = blockchain.add_block(bl1)
bl2 = Block('3', prev2)
prev3 = blockchain.add_block(bl2)
bl3 = Block('4', prev3)
prev4 = blockchain.add_block(bl3)

print(blockchain.check(prev4))

# Let's do a fork
bl4 = Block('88', prev)
prev4_1 = blockchain.add_block(bl4)

# See what's up:
print(blockchain.check(prev4_1))


#############
##EXAMPLE 2##
#############

blockchain_bueno = Blockchain()

genesis = Block('genesis', None)
hash_g = blockchain_bueno.add_block(genesis)
bl1 = Block('10', hash_g)
hash_1 = blockchain_bueno.add_block(bl1)
bl2 = Block('20', hash_1)
hash_2 = blockchain_bueno.add_block(bl2)
bl3 = Block('30', hash_2)
hash_3 = blockchain_bueno.add_block(bl3)

# First branch
bl3_1 = Block('30 - 1', hash_3)
hash_3_1 = blockchain_bueno.add_block(bl3_1)
bl4_1 = Block('40 - 1', hash_3_1)
hash_4_1 = blockchain_bueno.add_block(bl4_1)


# Second branch (i.e. first fork)
bl3_2 = Block('30 - 2', hash_3)
hash_3_2 = blockchain_bueno.add_block(bl3_2)
bl4_2 = Block('40 - 2', hash_3_2)
hash_4_2 = blockchain_bueno.add_block(bl4_2)

#blockchain_bueno.#()

# Test with the two heads
print(blockchain_bueno.check(hash_4_1))
print(blockchain_bueno.check(hash_4_2))


#############
##EXAMPLE 3##
#############


blockchain_malo = Blockchain()

genesis = Block('genesis', None)
hash_g = blockchain_malo.add_block(genesis)
bl1 = Block('10', hash_g)
hash_1 = blockchain_malo.add_block(bl1)
bl2 = Block('20', hash_1)
hash_2 = blockchain_malo.add_block(bl2)
bl3 = Block('30', hash_2)
hash_3 = blockchain_malo.add_block(bl3)

bl3_1 = Block('30 - 1', hash_3)
hash_3_1 = blockchain_malo.add_block(bl3_1)

# First tampering
bl3_1.data = '30 - 1 - corrupted'

# A good branch
bl3_2 = Block('30 - 2', hash_3)
hash_3_2 = blockchain_malo.add_block(bl3_2)

# Continue the bad branch
bl4_1 = Block('40 - 1', hash_3_1)
hash_4_1 = blockchain_malo.add_block(bl4_1)

# Continue the good branch
bl4_2 = Block('40 - 2', hash_3_2)
hash_4_2 = blockchain_malo.add_block(bl4_2)

#blockchain_malo.print_blockchain()

# Test with the two heads
print(blockchain_malo.check(hash_4_1)) # Bad
print(blockchain_malo.check(hash_4_2)) # Good

#############
##EXAMPLE 3##
#############


blockchain_malo_doble = Blockchain()

genesis = Block('genesis', None)
hash_g = blockchain_malo_doble.add_block(genesis)
bl1 = Block('10', hash_g)
hash_1 = blockchain_malo_doble.add_block(bl1)
bl2 = Block('20', hash_1)
hash_2 = blockchain_malo_doble.add_block(bl2)
bl3 = Block('30', hash_2)
hash_3 = blockchain_malo_doble.add_block(bl3)

bl3_1 = Block('30 - 1', hash_3)
hash_3_1 = blockchain_malo_doble.add_block(bl3_1)

# Branch 1 corrupted
bl3_1.data = '30 - 1 - corrupted'

bl3_2 = Block('30 - 2', hash_3)
hash_3_2 = blockchain_malo_doble.add_block(bl3_2)

bl4_1 = Block('40 - 1', hash_3_1)
hash_4_1 = blockchain_malo_doble.add_block(bl4_1)
bl4_2 = Block('40 - 2', hash_3_2)
hash_4_2 = blockchain_malo_doble.add_block(bl4_2)

# Branch 2 corrupted
bl4_2.data = '40 - 2 - corrupted'

bl5_2 = Block('50 - 2', hash_4_2)
hash_5_2 = blockchain_malo_doble.add_block(bl5_2)

#blockchain_malo_doble.print_blockchain()

# Test with the two heads
print(blockchain_malo_doble.check(hash_4_1))
print(blockchain_malo_doble.check(hash_5_2))

#############
##EXAMPLE 4##
#############


blockchain_malo_root = Blockchain()

genesis = Block('genesis', None)
hash_g = blockchain_malo_root.add_block(genesis)
bl1 = Block('10', hash_g)
hash_1 = blockchain_malo_root.add_block(bl1)
bl2 = Block('20', hash_1)
hash_2 = blockchain_malo_root.add_block(bl2)

# Branch root corrupted
bl2.data = '20 - corrupted'

bl3 = Block('30', hash_2)
hash_3 = blockchain_malo_root.add_block(bl3)

bl3_1 = Block('30 - 1', hash_3)
hash_3_1 = blockchain_malo_root.add_block(bl3_1)

bl3_2 = Block('30 - 2', hash_3)
hash_3_2 = blockchain_malo_root.add_block(bl3_2)

bl4_1 = Block('40 - 1', hash_3_1)
hash_4_1 = blockchain_malo_root.add_block(bl4_1)
bl4_2 = Block('40 - 2', hash_3_2)
hash_4_2 = blockchain_malo_root.add_block(bl4_2)

bl5_2 = Block('50 - 2', hash_4_2)
hash_5_2 = blockchain_malo_root.add_block(bl5_2)

#blockchain_malo_root.print_blockchain()

# Test with the two heads
print(blockchain_malo_root.check(hash_4_1))
print(blockchain_malo_root.check(hash_5_2))

