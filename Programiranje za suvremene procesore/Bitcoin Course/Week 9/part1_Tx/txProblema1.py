# As usual, most of the code is stolen from Jimmy Song's book, I just made it worse


from io import BytesIO
from unittest import TestCase

import json
import requests

from ecc import PrivateKey
from base58 import (
    encode_varint,
    hash256,
    int_to_little_endian,
    little_endian_to_int,
    read_varint,
    SIGHASH_ALL,
)
from scriptSimplified import Script


# This method connects to a full node to receive a transaction in order to later validate it
# The API we're using is a bit weird, but you can try with another one if you prefer
class TxFetcher:
    cache = {}

    @classmethod
    def get_url(cls, testnet=False):
        if testnet:
            return 'https://mempool.space/testnet/api/'
        else:
            return 'https://mempool.space/api/'

    @classmethod
    def fetch(cls, tx_id, testnet=False, fresh=False):
        if fresh or (tx_id not in cls.cache):
            url = '{}/tx/{}/hex'.format(cls.get_url(testnet), tx_id)
            response = requests.get(url)
            try:
                raw = bytes.fromhex(response.text.strip())
            except ValueError:
                raise ValueError('unexpected response: {}'.format(response.text))
            # make sure the tx we got matches to the hash we requested
            if raw[4] == 0:
                # zero imputs = coinbase
                raw = raw[:4] + raw[6:]
                tx = Tx.parse(BytesIO(raw), testnet=testnet)
                tx.locktime = little_endian_to_int(raw[-4:])
            else:
                tx = Tx.parse(BytesIO(raw), testnet=testnet)
#            if tx.id() != tx_id:
#                raise ValueError('not the same id: {} vs {}'.format(tx.id(), tx_id))
            cls.cache[tx_id] = tx
        cls.cache[tx_id].testnet = testnet
        return cls.cache[tx_id]

    @classmethod
    def load_cache(cls, filename):
        disk_cache = json.loads(open(filename, 'r').read())
        for k, raw_hex in disk_cache.items():
            raw = bytes.fromhex(raw_hex)
            if raw[4] == 0:
                raw = raw[:4] + raw[6:]
                tx = Tx.parse(BytesIO(raw))
                tx.locktime = little_endian_to_int(raw[-4:])
            else:
                tx = Tx.parse(BytesIO(raw))
            cls.cache[k] = tx

    @classmethod
    def dump_cache(cls, filename):
        with open(filename, 'w') as f:
            to_dump = {k: tx.serialize().hex() for k, tx in cls.cache.items()}
            s = json.dumps(to_dump, sort_keys=True, indent=4)
            f.write(s)

'''
# The transaction class
'''
class Tx:

    '''
    What defines a transaction:
    1. Version
    2. Locktime
    3. Inputs
    4. Outputs
    5. Network (testnet or not)
    '''
    def __init__(self, version, tx_ins, tx_outs, locktime, testnet=False):
        self.version = version
        self.tx_ins = tx_ins
        self.tx_outs = tx_outs
        self.locktime = locktime
        self.testnet = testnet

    # As usual, our prettyprint for printing the transaction; nothing interesting here
    def __repr__(self):
        tx_ins = ''
        for tx_in in self.tx_ins:
            tx_ins += tx_in.__repr__() + '\n'
        tx_outs = ''
        for tx_out in self.tx_outs:
            tx_outs += tx_out.__repr__() + '\n'
        return 'tx: {}\nversion: {}\ntx_ins:\n{}tx_outs:\n{}locktime: {}'.format(
            self.id(),
            self.version,
            tx_ins,
            tx_outs,
            self.locktime,
        )

    # Transaction ID; this is our hash pointer in the UTXO set
    # I.e. when you ask for a transaction to a full node, it guards it under this key
    def id(self):
        '''Human-readable hexadecimal of the transaction hash'''
        return self.hash().hex()

    # Well, just the hash
    # Note that the hash is given in "little-endian"
    def hash(self):
        '''Binary hash of the legacy serialization'''
        return hash256(self.serialize())[::-1]

    # When we receive raw transaction (bytes) we want to parse it to figure out what's what
    @classmethod
    def parse(cls, s, testnet=False):
        '''Takes a byte stream and parses the transaction at the start
        return a Tx object
        '''
        # s.read(n) will return n bytes
        # version is an integer in 4 bytes, little-endian
        version = little_endian_to_int(s.read(4))
        # num_inputs is a varint, use read_varint(s)
        num_inputs = read_varint(s)
        # parse num_inputs number of TxIns
        inputs = []
        for _ in range(num_inputs):
            inputs.append(TxIn.parse(s))
        # num_outputs is a varint, use read_varint(s)
        num_outputs = read_varint(s)
        # parse num_outputs number of TxOuts
        outputs = []
        for _ in range(num_outputs):
            outputs.append(TxOut.parse(s))
        # locktime is an integer in 4 bytes, little-endian
        locktime = little_endian_to_int(s.read(4))
        # return an instance of the class (see __init__ for args)
        return cls(version, inputs, outputs, locktime, testnet=testnet)

    # When we have a transaction object, to send it to the network we need to serialize it
    def serialize(self):
        '''Returns the byte serialization of the transaction'''
        # serialize version (4 bytes, little endian)
        result = int_to_little_endian(self.version, 4)
        # encode_varint on the number of inputs
        result += encode_varint(len(self.tx_ins))
        # iterate inputs
        for tx_in in self.tx_ins:
            # serialize each input
            result += tx_in.serialize()
        # encode_varint on the number of outputs
        result += encode_varint(len(self.tx_outs))
        # iterate outputs
        for tx_out in self.tx_outs:
            # serialize each output
            result += tx_out.serialize()
        # serialize locktime (4 bytes, little endian)
        result += int_to_little_endian(self.locktime, 4)
        return result

    # Computes the transaction fee
    # Recall that this will need to connect to a full node to get the input transaction!!!
    def fee(self):
        '''Returns the fee of this transaction in satoshi'''
        ######################
        ####IMPLEMENT THIS####
        ######################
        return 0



class TxIn:

    def __init__(self, prev_tx, prev_index, script_sig=None, sequence=0xffffffff):
        self.prev_tx = prev_tx
        self.prev_index = prev_index
        if script_sig is None:
            self.script_sig = Script()
        else:
            self.script_sig = script_sig
        self.sequence = sequence

    def __repr__(self):
        return '{}:{}'.format(
            self.prev_tx.hex(),
            self.prev_index,
        )

    @classmethod
    def parse(cls, s):
        '''Takes a byte stream and parses the tx_input at the start
        return a TxIn object
        '''
        # prev_tx is 32 bytes, little endian
        prev_tx = s.read(32)[::-1]
        # prev_index is an integer in 4 bytes, little endian
        prev_index = little_endian_to_int(s.read(4))
        # use Script.parse to get the ScriptSig
        script_sig = Script.parse(s)
        # sequence is an integer in 4 bytes, little-endian
        sequence = little_endian_to_int(s.read(4))
        # return an instance of the class (see __init__ for args)
        return cls(prev_tx, prev_index, script_sig, sequence)

    def serialize(self):
        '''Returns the byte serialization of the transaction input'''
        # serialize prev_tx, little endian
        result = self.prev_tx[::-1]
        # serialize prev_index, 4 bytes, little endian
        result += int_to_little_endian(self.prev_index, 4)
        # serialize the script_sig
        result += self.script_sig.serialize()
        # serialize sequence, 4 bytes, little endian
        result += int_to_little_endian(self.sequence, 4)
        return result

    def value(self, testnet=False):
        '''Get the outpoint value by looking up the tx hash
        Returns the amount in satoshi
        '''
        ######################
        ####IMPLEMENT THIS####
        ######################
        # You will have to use the TxFetcher class
        # For this, remember to provide the correct testnet parameter
        return 0


class TxOut:

    def __init__(self, amount, script_pubkey):
        self.amount = amount
        self.script_pubkey = script_pubkey

    def __repr__(self):
        return '{}:{}'.format(self.amount, self.script_pubkey)

    @classmethod
    def parse(cls, s):
        '''Takes a byte stream and parses the tx_output at the start
        return a TxOut object
        '''
        # amount is an integer in 8 bytes, little endian
        amount = little_endian_to_int(s.read(8))
        # use Script.parse to get the ScriptPubKey
        script_pubkey = Script.parse(s)
        # return an instance of the class (see __init__ for args)
        return cls(amount, script_pubkey)

    def serialize(self):
        '''Returns the byte serialization of the transaction output'''
        # serialize amount, 8 bytes, little endian
        result = int_to_little_endian(self.amount, 8)
        # serialize the script_pubkey
        result += self.script_pubkey.serialize()
        return result

'''
raw_tx = bytes.fromhex('0100000001813f79011acb80925dfe69b3def355fe914bd1d96a3f5f71bf8303c6a989c7d1000000006b483045022100ed81ff192e75a3fd2304004dcadb746fa5e24c5031ccfcf21320b0277457c98f02207a986d955c6e0cb35d446a89d3f56100f4d7f67801c31967743a9c8e10615bed01210349fc4e631e3624a545de3f89f5d8684c7b8138bd94bdd531d2e213bf016b278afeffffff02a135ef01000000001976a914bc3b654dca7e56b04dca18f2566cdaf02e8d9ada88ac99c39800000000001976a9141c4bc762dd5423e332166702cb75f40df79fea1288ac19430600')
stream = BytesIO(raw_tx)
tx = Tx.parse(stream)

print(tx)
'''

'''
######################
########TEST1#########
######################
raw_tx = bytes.fromhex('0100000001813f79011acb80925dfe69b3def355fe914bd1d96a3f5f71bf8303c6a989c7d1000000006b483045022100ed81ff192e75a3fd2304004dcadb746fa5e24c5031ccfcf21320b0277457c98f02207a986d955c6e0cb35d446a89d3f56100f4d7f67801c31967743a9c8e10615bed01210349fc4e631e3624a545de3f89f5d8684c7b8138bd94bdd531d2e213bf016b278afeffffff02a135ef01000000001976a914bc3b654dca7e56b04dca18f2566cdaf02e8d9ada88ac99c39800000000001976a9141c4bc762dd5423e332166702cb75f40df79fea1288ac19430600')
stream = BytesIO(raw_tx)
tx = Tx.parse(stream)

print('Output1:',tx.fee())
'''

'''
######################
########TEST2#########
######################

# This uses another API/node just for checking stuff

# The generic format (for mainnet):
#https://blockchain.info/rawtx/b6f6991d03df0e2e04dafffcd6bc418aac66049e2cd74b80f14ac86db1e3f0da?format=hex

#mainnet transaction:
tx_hash = 'b6f6991d03df0e2e04dafffcd6bc418aac66049e2cd74b80f14ac86db1e3f0da'

url = 'https://blockchain.info/rawtx/{}?format=hex'.format(tx_hash)
response = requests.get(url)


raw = bytes.fromhex(response.text)
stream = BytesIO(raw)
tx = Tx.parse(stream)

print('Output2:',tx.fee())


'''

'''
######################
########TEST3#########
######################

#####################################################
###### One of the homework2 transactions: ###########
#####################################################
tx_hash = '15e49ac9d29766cfbc73bfd89d3cab5fbf7ee6eff015553b1977cf0a9b68c4ae'
fetcher = TxFetcher()
tx = fetcher.fetch(tx_hash,testnet=True)

print('Output3:',tx.fee())
'''