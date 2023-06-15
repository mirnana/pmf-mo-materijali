
from bloomfilter import BloomFilter
from helper import decode_base58
from merkleblock import MerkleBlock
from network import FILTERED_BLOCK_DATA_TYPE, GetHeadersMessage, GetDataMessage, HeadersMessage, SimpleNode
from tx import Tx


# last block we know:
last_block_hex = '00000000000002e5fc775089469c567efc54879bd23172edcdda29f9f0242342'
# stuff we're looking for (it's in block 25):
address = 'n3jKhCmVjvaVgg8C5P7E48fdRkQAAvf7Wc'
h160 = decode_base58(address)


# Establish a connection to a testnet node#
node = SimpleNode('testnet.programmingbitcoin.com', testnet=True, logging=False)
# Define our bloom filter
bf = BloomFilter(size=30, function_count=5, tweak=90210)
# Put the data into the filter
bf.add(h160) 
# Handshake and load the filter onto the connection
node.handshake()
node.send(bf.filterload()) 

# Get block headers (2000 starting from last_block_hex)
start_block = bytes.fromhex(last_block_hex)
getheaders = GetHeadersMessage(start_block=start_block)
node.send(getheaders) 
headers = node.wait_for(HeadersMessage)

# Load a get data message with this stuff
getdata = GetDataMessage() 
for b in headers.blocks:
	if not b.check_pow():
		raise RuntimeError('proof of work is invalid')
	getdata.add_data(FILTERED_BLOCK_DATA_TYPE, b.hash())

# Ask for data in these headers
node.send(getdata)

# The node replying to this message will send:
# 1. A MerkleBlock with:
#		- A Merkle Proof when a tx matches the filter
#		- With empty Merkle Proof otherwise (just the root)
# 2. A Tx message if any of the Txs in the block matches the filter

# Wait for the data; this is a bit poorly implemented
# We have 2000 hashes above to run through, so we'll exhaust them
# The breaking conditions need work, but this is just to demo ;-)
# Namely, I know that we will receive 2003 messages; should probably wait for silence on the channel between messages
j = 2003
while j>0:
    message = node.wait_for(MerkleBlock, Tx)
    j = j - 1
    # A mekleblock message that matches the filter will send the proof as well
    # The one that does not will just send a single hash proof (the root)
    if message.command == b'merkleblock':
        #print('block:', message.hash().hex())
        #if (len(message.hashes)>1):
        #for x in message.hashes:
        #	print(x.hex()[::-1])
    	if not message.is_valid():
            raise RuntimeError('invalid merkle proof')
    # Here we check if output matches our address
    # In this case we print it out
    else:
    	for i, tx_out in enumerate(message.tx_outs):
            if tx_out.script_pubkey.address(testnet=True) == address: 
                print('found: {}:{}'.format(message.id(), i))
