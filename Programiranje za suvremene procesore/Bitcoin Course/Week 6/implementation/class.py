from ecc import *
from base58 import *
import json



# The objective of the class will be to implement some aspects of ScroogeCoin
# In particular, we will implement transactions from the end user viewpoint
# Meaning we will be able to form transactions, sign them, and verify them
# We will also implement an UTXO pool that Scrooge uses to check whether a tranaction is valid




# Let's Define who the boss is
privKeyScrooge = PrivateKey(256)
pubKeyScrooge = privKeyScrooge.point
# Obviously this is a dumb PrivateKey and you should never use something like this
# All things have to be validated by Scrooge
# All coins need to be created by Scrooge
# So he will sign a lot of stuff





# The first thing we want to define is transactions
# Recall the components of transactions: ID, type, inputs, outputs, and signatures
# We will begin by defining inputs and outputs



'''
# To define an input for a transaction we need four things:
	1) ID of the transaction where the coins were created (i.e. where these inputs are outputs)
	2) The output number for this transaction (recall coinID 73(32) -- 73 is txID, and 32 output)
	3) The value
	4) The owner

Let's code this:
'''


class Input:
	def __init__(self, txID, nrOutput, amount, address):
		self.whereCreated = txID
		self.nrOutput = nrOutput
		self.value = amount
		self.owner = address #raw public key (as a curve point object)

	#for hashing an input
	def serialize(self):
		d = { 'whereCreated': self.whereCreated,
		      'output_number': self.nrOutput,
		      'value': self.value,
		      'owner': str(self.owner) }
			
		return json.dumps(d, sort_keys=True).encode('utf-8')

	# for printing and debugging
	def __str__(self):
		return '(Origin tx: {} - Origin output: {} - Value: {} - Owner: {})'.\
			format(self.whereCreated,self.nrOutput,self.value,str(self.owner))



'''
The next class will represent transaction outputs. These are defined by:
	1) The value they have
	2) The owner of this output

The txID and output number will be implicit from the context where we define the outputs
'''



class Output:
	def __init__(self, amount, address):
		self.value = amount
		self.recipient = address #raw public key

	#for hashing an output
	def serialize(self):
		d = { 'value': self.value,
		      'recipient': str(self.recipient) }
			
		return json.dumps(d, sort_keys=True).encode('utf-8')		

	# for printing
	def __str__(self):
		return '(Value: {} - Recipient: {})'.format(self.value,str(self.recipient))






'''
The next class will be the transaction class

A tranasaction is defined by:
	1) ID
	2) type (createCoins or payCoins)
	3) inputs (a list of objects of class Input)
	4) outputs (a list of objects of class Output)
	5) signatures of **all** input owners)

A quick note: the ID should be assigned by Scrooge in our system; i.e. he receives a transaction
without the ID, and signs it to put in a block. But we're just trying to highlight some details
here, not the whole system. One of possible projects will be to build Scrooge coin completely
'''




class Transaction:
	def __init__(self, txID, txType, inputCoins, outputCoins):
		self.txID = txID
		self.type = txType
		self.inputs = inputCoins
		self.outputs = outputCoins
		
		self.dataForSigs = self.DataForSigs() 
		# this is the data we will sign (serialization of the above)
		
		self.signatures = {} 
		# raw signatures (as a ECC signature object); 
		# stored in a  dictionary ['str(pubKeyX)'] = sig for pubkeyX
		# really we should have der signatures here, but since this would solve your homework ...
		# also, we should probably index by hash256(SEC(PubKey)).hex(), but no one is perfect


	def DataForSigs(self):

		rawInputs = ''

		for coin in self.inputs:
			rawInputs = rawInputs + str(coin.serialize())

		rawOutputs = ''

		for coin in self.outputs:
			rawOutputs = rawOutputs + str(coin.serialize())

		rawData = {'txID': self.txID,
					'type': str(self.type),
					'inputs': rawInputs,
					'outputs': rawOutputs }					

		# recall what I'm signing: hash of the message;
		# which is the same as a scalar in the group of my curve (as an int)
		return int(hash(bytes(json.dumps(rawData, sort_keys=True).encode('utf-8'))).hex(),16)


	# We will now implement a function that we need here
	# But first, let us play with what we have so far (I'll skip forward)
	

	# checks that all the signatures match up
	def CheckSignatures(self):
		toSign = self.dataForSigs

		if (self.type == 'payCoins'):

			if (len(self.inputs) == 0):
				return False

			for x in self.inputs:
				if (str(x.owner) not in self.signatures.keys()):
					return False
				else:
					if (not x.owner.verify(toSign,self.signatures[str(x.owner)])):
						return False

			return True

		if (self.type == 'createCoins'):
			if (len(self.signatures) > 1):				
				return False

			if (str(pubKeyScrooge) not in self.signatures.keys()):
				return False

			if (not pubKeyScrooge.verify(toSign,self.signatures[str(pubKeyScrooge)])):
				return False

			return True

		return False



	# checks that all the input values < output values
	def CheckValues(self):
		inValue = 0
		for x in self.inputs:
			if (x.value < 0):
				return False
			else:
				inValue += x.value

		outValue = 0
		for x in self.outputs:
			if (x.value < 0):
				return False
			else:
				outValue += x.value
				
		return (inValue >= outValue)


	def serialize(self):

		rawInputs = []

		for coin in self.inputs:
			rawInputs.append(str(coin.serialize()))

		rawOutputs = []

		for coin in self.outputs:
			rawOutputs.append(str(coin.serialize()))

		sigs = []

		for sig in self.signatures:
			sigs.append(str(self.signatures[sig]))

		rawData = { 'txID': self.txID,
					'type': str(self.type),
					'inputs': rawInputs,
					'outputs': rawOutputs,
					'signatures': sigs}					

		return json.dumps(rawData, sort_keys=True).encode('utf-8')		



pkA = PrivateKey(100)
pkB = PrivateKey(200)
pkC = PrivateKey(300)

addressA = pkA.point
addressB = pkB.point
addressC = pkC.point

inputs = []

input0 = Input(1,0,7.35,addressA)
input1 = Input(2,1,4.12,addressB)

inputs.append(input0)
inputs.append(input1)

outputs = []

out0 = Output(11.46,addressA)
outputs.append(out0)	


trans = Transaction(101,'payCoins',inputs,outputs)

#print(trans.serialize())

toSign = trans.dataForSigs

# Alice signs:
sigAlice = pkA.sign(toSign)
# Bob signs:
sigBob = pkB.sign(toSign)

# they complete the transaction:

trans.signatures[str(addressA)] = sigAlice
trans.signatures[str(addressB)] = sigBob

#print(trans.serialize())


# Is alice's signatureOK:
#print(addressA.verify(toSign,sigAlice))

# where is Alice's signature: sigs[str(addressA)]

'''
sigsOK = trans.CheckSignatures()
print(sigsOK)

checkValue = trans.CheckValues()
print(checkValue)
'''



'''
generating test data
'''

'''
##################
### EXAMPLE 1: ###
##################

inputs = []

input0 = Input(1,0,1,addressB)
input1 = Input(1,1,2,addressB)

inputs.append(input0)
inputs.append(input1)

outputs = []

out0 = Output(3,addressB)
outputs.append(out0)	


trans = Transaction(101,'payCoins',inputs,outputs)
toSign = trans.dataForSigs

sigBob = pkB.sign(toSign)
trans.signatures[str(addressB)] = sigBob


# they complete the transaction:
#trans.signatures[str(addressB)] = Signature(0x2158c3ea6df90ab39dcefb3db485bcaaf095a801914114cfcbc2756e89aa6a39,0xb970ae6b716da4261847176318aeab5bb9a34d24eeb1353c595005b9fd771bc)
#trans.signatures[str(addressB)] = Signature(0x2258c3ea6df90ab39dcefb3db485bcaaf095a801914114cfcbc2756e89aa6a39,0xb970ae6b716da4261847176318aeab5bb9a34d24eeb1353c595005b9fd771bc)

print('Example 1: ', trans.CheckSignatures())

##################
### EXAMPLE 2: ###
##################

inputs = []

input0 = Input(1,0,1,addressB)
input1 = Input(1,1,2,addressB)

inputs.append(input0)
inputs.append(input1)

outputs = []

out0 = Output(3,addressB)
outputs.append(out0)	


trans = Transaction(101,'payCoins',inputs,outputs)
toSign = trans.dataForSigs


# they complete the transaction:
trans.signatures[str(addressB)] = Signature(0x2258c3ea6df90ab39dcefb3db485bcaaf095a801914114cfcbc2756e89aa6a39,0xb970ae6b716da4261847176318aeab5bb9a34d24eeb1353c595005b9fd771bc)

print('Example 2: ', trans.CheckSignatures())
'''

##################
### EXAMPLE 3: ###
##################

inputs = []

input0 = Input(1,0,1,addressB)
input1 = Input(1,1,2,addressA)

inputs.append(input0)
inputs.append(input1)

outputs = []

out0 = Output(3,addressB)
outputs.append(out0)	


trans = Transaction(101,'payCoins',inputs,outputs)
toSign = trans.dataForSigs


# they complete the transaction:
trans.signatures[str(addressA)] = Signature(0x604b61b126dda950f6d4bc8d5d89552f14e5bdb108fe301482042e0fe71975a1,0x2158b0e795ea84c79cba8eb4ebd7d76658472486cbc9890afeffad79c2975bbe)
trans.signatures[str(addressB)] = Signature(0xbe6f90eb5f2bfdd0aa487afe05d17bd709b461e080633a65161f2508264f324e,0x3f24d15b0e00116e91c5f840ee9ba9be34bd889adcb55c745b39cf5ca2c94242)


print('Example 3: ', trans.CheckSignatures())
'''


##################
### EXAMPLE 4: ###
##################

inputs = []

input0 = Input(1,0,1,addressB)
input1 = Input(1,1,2,addressA)

inputs.append(input0)
inputs.append(input1)

outputs = []

out0 = Output(3,addressB)
outputs.append(out0)	


trans = Transaction(101,'payCoins',inputs,outputs)
toSign = trans.dataForSigs


# they complete the transaction:
trans.signatures[str(addressA)] = Signature(0x604b61b126dda950f6d4bc8d5d89552f14e5bdb108fe301482042e0fe71975a1,0x2158b0e795ea84c79cba8eb4ebd7d76658472486cbc9890afeffad79c2975bbe)
trans.signatures[str(addressB)] = Signature(0xbe6f90eb5f2b8dd0aa487afe05d17bd709b461e080633a65161f2508264f324e,0x3f24d15b0e00116e91c5f840ee9ba9be34bd889adcb55c745b39cf5ca2c94242)


print('Example 4: ', trans.CheckSignatures())


##################
### EXAMPLE 5: ###
##################

inputs = []

outputs = []

out0 = Output(3,addressA)
outputs.append(out0)	


trans = Transaction(101,'createCoins',inputs,outputs)

# they complete the transaction:
trans.signatures[str(pubKeyScrooge)] = Signature(0xf74c0eba31d119b4259d86d5c47c332f0df0e08d0685ec99882e65187407d7c8,0x7ed3d1476a446359387702d5ef819ee8e0a126e279c8dbeded6127dcde84ff74)


print('Example 5: ', trans.CheckSignatures())

'''



# The next objective we have is to see how Scrooge manages his UTXO pool
# Recall: Scrooge does not check the blockchain
# He instead maintains a pool of transactions that are still valid
# When a new transaction arrives he just checks the sigs, the values;
#	and that the input transactions are in his UTXO pool
# If they are, the transaction is valid
# To process the transaction and publish it on the blockchain, scrooge has to:
#	1) Remove all the inputs from his UTXO pool
#	2) Add all the outputs to his UTXO pool
# This is what we will try to simulate here
# Note that this is precisely what a Bitcoin full node does





# What does an UTXO pool contain? Just some transaction Outputs
# Recall that an Output of one transaction is an input to another transaction
# Also recall that an object of class Input has a bit more data than Output
# When we put an Output to an UTXO pool, we will already format it as input



# An UTXO pool simply stores the transactions in a dictionary called pool
# items are stored as pool[hash(Input).hex()] = Input
# The pool needs to process each transaction it receives



class UTXO_Pool:
	def __init__(self):
		# contains pairs (hash of output/input (as hex), output/input -- saved as valid UTXOs)
		self.pool = {}

	def processTransaction(self, transaction):

		# We first check that transaction signatures match up:
		if (not transaction.CheckSignatures()):
			return False
		# Note that this already makes any createCoins transaction valid!		

		# Next we will check when a payCoins type transaction is valid
		# Meaning the values are correct, and the inputs have not been spent yet
		if (transaction.type != 'createCoins'):

			# Checking that input values are higher than output values
			if (not transaction.CheckValues()):
				return False

			# Check that each input is correct
			for x in transaction.inputs:				
				if (not self.validInput(x)):					
					return False
				
			# If we made it this far, we know that the transaction is valid
			# so we can update the UTXO pool by deleting the inputs from the pool
			for x in transaction.inputs:								
				toDelete = hash(x.serialize()).hex()
				self.pool.pop(toDelete)

		# If we made it this far, we know that the transaction is good
		# So we will add it's outputs to the UTXO pool

		# i stores the position of the current output
		i = 0
		for x in transaction.outputs:
			# We already prepare the output to be an input for the next transaction
			utxo = Input(transaction.txID,i,x.value,x.recipient)
			# Next we add this output/input to the pool with ts key being the hash
			utxoHash = hash(utxo.serialize()).hex()
			self.pool[utxoHash] = utxo
			# we increase the output counter
			i += 1



	# This method checks that the transaction input is valid
	def validInput(self,input):		

		# First we will verify that the Input is in our UTXO pool
		# Recall that we index our UTXO pool by hashes
		toHash = input.serialize()

		inputHash = hash(toHash).hex()

		# If we do not find the Input in the pool, the Input is not valid
		if (inputHash not in self.pool.keys()):
			return False

		# We also do a sanity check that the input was not changed when creating the transaction
		# This means that the input is the same as the one recoreded on the blockchain
		# For this the four components of the input must be identical

		# Check that the transaction ID where the output/input was defined is correct
		if (self.pool[inputHash].whereCreated != input.whereCreated):
			return False

		# Check that in coinID(x) x is specified correctly
		if (self.pool[inputHash].nrOutput != input.nrOutput):
			return False			

		# Check that someone is not trying to claim more coins than they actually have here
		if (self.pool[inputHash].value != input.value):
			return False			

		# Check that the address of the owner is correct
		if (self.pool[inputHash].owner != input.owner):
			return False

		# If all the checks pass, we are good
		return True



#Let's now see how the system operates as seen from Scrooges perspective


'''
# upool will be our UTXO pool
upool = UTXO_Pool()

###################
###TRANSACTION 1###
###################

#initialize the system with scrooge giving out money
outputs = [Output(10,addressA),Output(10,addressB),Output(10,addressC)]
trans1 = Transaction(1,'createCoins',[],outputs)
to_sign = trans1.dataForSigs

trans1.signatures[str(pubKeyScrooge)] = privKeyScrooge.sign(to_sign)


upool.processTransaction(trans1)

print('What the UTXO pool looks like after transaction 1:\n')
for x in upool.pool:
	print(upool.pool[x],'\n')







###################
###TRANSACTION 2###
###################


inputs = [Input(1,0,10,addressA),Input(1,1,10,addressB),Input(1,2,10,addressC)]

outputs = [Output(30,addressA)]

trans2 = Transaction(2,'payCoins',inputs,outputs)
toSign = trans2.dataForSigs
	
trans2.signatures[str(addressA)] = pkA.sign(toSign)
trans2.signatures[str(addressB)] = pkB.sign(toSign)
trans2.signatures[str(addressC)] = pkC.sign(toSign)

upool.processTransaction(trans2)

print('What the UTXO pool looks like after transaction 2:\n')
for x in upool.pool:
	print(upool.pool[x],'\n')








###################
###TRANSACTION 3###
###################


# Let's see if there are any changes when the trasaction does something bad

inputs = [Input(2,0,30,addressA)]

outputs = [Output(30,addressB)]

trans3 = Transaction(3,'payCoins',inputs,outputs)
toSign = trans3.dataForSigs

# Bob is trying to fake Alice's signature	
trans3.signatures[str(addressA)] = pkB.sign(toSign)

upool.processTransaction(trans3)

print('What the UTXO pool looks like after transaction 3:\n')
for x in upool.pool:
	print(upool.pool[x],'\n')








###################
###TRANSACTION 4###
###################


# Let's see if there are any changes when the trasaction does something bad

inputs = [Input(1,0,10,addressA)]

outputs = [Output(10,addressB)]

trans4 = Transaction(4,'payCoins',inputs,outputs)
toSign = trans4.dataForSigs

# Alice is trying to spend an already spent transaction output
trans4.signatures[str(addressA)] = pkA.sign(toSign)

upool.processTransaction(trans4)

print('What the UTXO pool looks like after transaction 4:\n')
for x in upool.pool:
	print(upool.pool[x],'\n')




###################
###TRANSACTION 5###
###################


# Let's spend Alice's money

inputs = [Input(2,0,30,addressA)]

outputs = [Output(10,addressB),Output(20,addressA)]

trans5 = Transaction(5,'payCoins',inputs,outputs)
toSign = trans5.dataForSigs

trans5.signatures[str(addressA)] = pkA.sign(toSign)

upool.processTransaction(trans5)

print('What the UTXO pool looks like after transaction 5:\n')
for x in upool.pool:
	print(upool.pool[x],'\n')

'''

# That's how a full node processes blocks
# The only difference is that each block can have more than one transaction
# A fun exercize is to try and implement that
