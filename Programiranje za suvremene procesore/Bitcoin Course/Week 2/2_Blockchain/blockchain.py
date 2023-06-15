from hash import hash
import json

class Block:
	def __init__(self, data):
		self.data = data
		self.prev_hash = None

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
		self.head = None

	def add_block(self, data):
		block = Block(data)

		block.prev_hash = self.head # first head is None by default

		serialize_to_hash = hash(block.serialize()).hex() # compute the new key/i.e. the new hash pointer to the head
		self.elements[serialize_to_hash] = block # insert the data in the dictionary under the computed key
		self.head = serialize_to_hash # make the new hash pointer the head of the blockchain

	def get_head(self):
		return self.head

	def print_blockchain(self):
		for key in self.elements.keys():
			print('Key: {}'.format(key))
			print('Block: {}'.format(self.elements[key]))

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

#############
##EXAMPLE 1##
#############


blockchain = Blockchain()
blockchain.add_block('one')
blockchain.add_block('42')
blockchain.add_block('three')
blockchain.add_block('44')

#blockchain.print_blockchain()

x = blockchain.get_head()
print(blockchain.check(x))

blockchain.elements[blockchain.elements[x].prev_hash].data = '1'

print(blockchain.check(x))


#############
##EXAMPLE 2##
#############

blockchain_bueno = Blockchain()

blockchain_bueno.add_block('10')
blockchain_bueno.add_block('20')
blockchain_bueno.add_block('30')
blockchain_bueno.add_block('40')
blockchain_bueno.add_block('50')

#blockchain_bueno.print_blockchain()

# Test with the third block
print(blockchain_bueno.check('83e18321fe0a34e69c94288d9d725cb9340dcf421db446bbf155d65a47956201'))

#############
##EXAMPLE 3##
#############


blockchain_malo = Blockchain()

blockchain_malo.add_block('11')
blockchain_malo.add_block('22')
blockchain_malo.add_block('33')
blockchain_malo.add_block('44')
blockchain_malo.add_block('55')

# Modifying the data of the third block
blockchain_malo.elements['770ca53c5b6d7d72ece11f56d53706331a7ffe5a2d67f81d7b97375c4d0e5146'].data = 34

#blockchain_malo.print_blockchain()

# Test with the second and third block
print(blockchain_malo.check('dc92a16bc557bbddf0e16e0cbe341f1c7f08c5187b8b20338225be8c89816e9c'))
print(blockchain_malo.check('770ca53c5b6d7d72ece11f56d53706331a7ffe5a2d67f81d7b97375c4d0e5146'))

# test from head of thechain:
print(blockchain_malo.check('7d042ebae77f18ef5c9c48de337a12a86c141007fa0bbe4a003851b37cd7a5d7'))


#############
##EXAMPLE 4##
#############

blockchain_key_error = Blockchain()

blockchain_key_error.add_block('12')
blockchain_key_error.add_block('24')
blockchain_key_error.add_block('36')
blockchain_key_error.add_block('48')
blockchain_key_error.add_block('60')
blockchain_key_error.add_block('72')

#blockchain_key_error.print_blockchain()

blockchain_key_error.elements['a12349efcd4407932996a44dd56fd136710961314aaf29fe2f1ab7cf36adb1c1'].prev_hash = 'wrong key'

# Test with the head and the third block
print(blockchain_key_error.check('01d74f2fdee3805f55fdf6f767cf3ca61e53caf908fc4ae4805136c35cb2edea'))
print(blockchain_key_error.check('79203ae2f1b5ef3a70e93906be26a5f0540f0ad35adac5239dcf1350bcb07fd6'))


	
