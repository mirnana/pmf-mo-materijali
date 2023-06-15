from hash import *
from random import randint
import hmac

# The objective of this activity is to implement ECDSA as used in BitCoin
# We will basically transform the theory from the last two lectures into Python code
# At the end we should be able to produce raw signatures as used in BitCoin


# First, we will code finite fields
# As with curves, we do not need an entire field, just its individual elements

class FieldElement:

    def __init__(self, num, prime):
        # Finite field is defined by its order p (i.e. the prime number)
        # An element is just a number in this range
        if num >= prime or num < 0:
            error = 'Num {} not in field range 0 to {}'.format(
                num, prime - 1)
            raise ValueError(error)
            # strictly speaking, we could have just done % prime to put the number in the correct range
        self.num = num
        self.prime = prime

    def __repr__(self):
        # for printing
        return 'FieldElement_{}({})'.format(self.prime, self.num)

    def __eq__(self, other):
        # allows us to do equality comparisons with field elements
        if other is None:
            return False
        return self.num == other.num and self.prime == other.prime

    def __ne__(self, other):
        # this should be the inverse of the == operator
        return not (self == other)

    def __add__(self, other):
        # adds two elements of the field
        if self.prime != other.prime:
            raise TypeError('Cannot add two numbers in different Fields')
        # self.num and other.num are the actual values
        # self.prime is what we need to mod against
        num = (self.num + other.num) % self.prime
        # We return an element of the same class
        return self.__class__(num, self.prime)

    def __sub__(self, other):
        # subtracts two elements of the field
        if self.prime != other.prime:
            raise TypeError('Cannot subtract two numbers in different Fields')
        # self.num and other.num are the actual values
        # self.prime is what we need to mod against
        num = (self.num - other.num) % self.prime
        # We return an element of the same class
        return self.__class__(num, self.prime)

    def __mul__(self, other):
        # Multiplication in F_p
        if self.prime != other.prime:
            raise TypeError('Cannot multiply two numbers in different Fields')
        # self.num and other.num are the actual values
        # self.prime is what we need to mod against
        num = (self.num * other.num) % self.prime
        # We return an element of the same class
        return self.__class__(num, self.prime)

    def __pow__(self, exponent):
        # Exponentiation
        n = exponent % (self.prime - 1)
        # the real exponent is computed using Fermat's little theorem
        num = pow(self.num, n, self.prime)
        return self.__class__(num, self.prime)

    def __rmul__(self, coefficient):
        # scalar product in F_p
        num = (self.num * coefficient) % self.prime
        return self.__class__(num=num, prime=self.prime)

    def __truediv__(self, other):
        # division of two elements in F_p
        if self.prime != other.prime:
            raise TypeError('Cannot divide two numbers in different Fields')
        # self.num and other.num are the actual values
        # self.prime is what we need to mod against
        # use fermat's little theorem:
        # self.num**(p-1) % p == 1
        # this means:
        # 1/n == pow(n, p-2, p)
        num = (self.num * pow(other.num, self.prime - 2, self.prime)) % self.prime
        # We return an element of the same class
        return self.__class__(num, self.prime)

'''
element1 = FieldElement(3,47)

print("element1 = ", element1)

element2 = FieldElement(46,47)

print("element1 = ", element2)

sum = element1 + element2

print("Sum: ", sum)

fraction = element1 / element2

print("Division: ", fraction)

print(element1 == element2)
print(element1 != element2)
'''




# The next class we will need is a elliptic curve point
# The point is defined by its coordinates AND the underlying curve
# I.e. (2,3) on one curve is not the same thing as (2,3) on another curve
# Curve equation y**2 = x**3 + a*x + b
# So (a,b) are enought to know which curve we are working on


class Point:

    def __init__(self, x, y, a, b):
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        # x being None and y being None represents the point at infinity
        # Check for that here since the equation below won't make sense
        # with None values for both.
        if self.x is None and self.y is None:
            return
        # make sure that the elliptic curve equation is satisfied
        # y**2 == x**3 + a*x + b
        if self.y**2 != self.x**3 + a * x + b:
            # if not, throw a ValueError
            raise ValueError('({}, {}) is not on the curve'.format(x, y))

    def __eq__(self, other):
        # Check if two points are equal; for this they have to be on the same curve
        return self.x == other.x and self.y == other.y \
            and self.a == other.a and self.b == other.b

    def __ne__(self, other):
        # this should be the inverse of the == operator
        return not (self == other)

    def __repr__(self):
        # For printing purposes
        if self.x is None:
            return 'Point(infinity)'
        elif isinstance(self.x, FieldElement):
            # when we're over a finite field
            return 'Point({},{})_{}_{} FieldElement({})'.format(
                self.x.num, self.y.num, self.a.num, self.b.num, self.x.prime)
        else:
            return 'Point({},{})_{}_{}'.format(self.x, self.y, self.a, self.b)

    def __add__(self, other):
        # Add two pints; the returned element should be an instance of the class Point
        if self.a != other.a or self.b != other.b:
            raise TypeError('Points {}, {} are not on the same curve'.format(self, other))
        # Case 0.0: self is the point at infinity, return other
        if self.x is None:
            return other
        # Case 0.1: other is the point at infinity, return self
        if other.x is None:
            return self

        # Case 1: self.x == other.x, self.y != other.y
        # Result is point at infinity
        if self.x == other.x and self.y != other.y:
            return self.__class__(None, None, self.a, self.b)

        # Case 2: self.x ≠ other.x
        # Formula (x3,y3)==(x1,y1)+(x2,y2)
        # s=(y2-y1)/(x2-x1)
        # x3=s**2-x1-x2
        # y3=s*(x1-x3)-y1
        if self.x != other.x:
            s = (other.y - self.y) / (other.x - self.x)
            x = s**2 - self.x - other.x
            y = s * (self.x - x) - self.y
            return self.__class__(x, y, self.a, self.b)

        # Case 4: if we are tangent to the vertical line,
        # we return the point at infinity
        # note instead of figuring out what 0 is for each type
        # we just use 0 * self.x
        if self == other and self.y == 0 * self.x:
            return self.__class__(None, None, self.a, self.b)

        # Case 3: self == other
        # Formula (x3,y3)=(x1,y1)+(x1,y1)
        # s=(3*x1**2+a)/(2*y1)
        # x3=s**2-2*x1
        # y3=s*(x1-x3)-y1
        if self == other:
            s = (3 * self.x**2 + self.a) / (2 * self.y)
            x = s**2 - 2 * self.x
            y = s * (self.x - x) - self.y
            return self.__class__(x, y, self.a, self.b)

    def __rmul__(self, coefficient):
        # Scalar multiplication e*P
        # We will use the double and add method
        coef = coefficient
        current = self
        result = self.__class__(None, None, self.a, self.b)
        while coef:
            if coef & 1:
                result += current
            current += current
            coef >>= 1
        return result


'''
a = 0
b = 17


p1 = Point(2,5,a,b)

print("p1 = ", p1)

p2 = Point(4,9,a,b)

print("p2 = ", p2)

p3 = Point(4,-9,a,b)

print("-p2 = ", p3)

print("Sum of two different elements p1 + p2 =  ", p1 + p2)

print("Sum of the element with it inverse p2 - p2 =  ", p2 + p3)

print("Sum of the element with itself:  ", p1 + p1)
# Why did this not work??
# Because computers suck at continuous math
# So we do what we know how to do ... goto finite fields


'''






'''

#a = 0
#b = 7
# This will not work if we're over a finite field

p = 223
a = FieldElement(0,p)
b = FieldElement(7,p)

x = FieldElement(170,p)
y = FieldElement(142,p)

p1 = Point(x,y,a,b)

print("Point p1 = ",p1)

print("Adding a point to itself p1 + p1 = ", p1+p1)
print("Scalar multiplication 2*p1 = ", 2*p1)
print("Scalar multiplication 1234567890*p1 = ", 1234567890*p1)

'''




# We will be working with BitCoin's curve and finite field all the time, so let's make life easier for us
# Meaning that the Field Order will be implicit, so we just define the point through its coordinates



# a and b of secp256k1
A = 0
B = 7
# field order
P = 2**256 - 2**32 - 977
# subgroup order for secp256k1; will not be used for much
N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141


# instantiate the finite field needed for secp256k1
# The field order (not to be confused with the subgroup order!!!) will be implicit
class S256Field(FieldElement):

    def __init__(self, num, prime=None):
        super().__init__(num=num, prime=P)

    def __repr__(self):
        return '{:x}'.format(self.num).zfill(64)


'''
p = S256Field(7)
print(p)


# Don't get confused by the format:
p = S256Field(10)
print(p)
'''



# point on secp256k1 curve
# This is used so that we don't have to write FieldElement(x,p) all the time
class S256Point(Point):

    def __init__(self, x, y, a=None, b=None):
        a, b = S256Field(A), S256Field(B)
        if type(x) == int:
            super().__init__(x=S256Field(x), y=S256Field(y), a=a, b=b)
        else:
            super().__init__(x=x, y=y, a=a, b=b)  # point at infinity

    def __repr__(self):
        if self.x is None:
            return 'S256Point(infinity)'
        else:
            return 'S256Point({}, {})'.format(self.x, self.y)

# speeding up the scalar multiplication by modding with the group order
    def __rmul__(self, coefficient):
        coef = coefficient % N  # mod by group order does the same thing
        return super().__rmul__(coef)


'''
# Now specifying points is much simpler for us

v = S256Field(8)

w = v**((P+1)//4) # We have to compute the y coordinate

p = S256Point(1,w.num)
print(p)
'''


# secp256k1 subgroup generator
G = S256Point(
    0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
    0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)




# the signature class
# we just need to return the pair (r,s) -- this is non serialized signature
class Signature:

    def __init__(self, r, s):
        self.r = r
        self.s = s

    def __repr__(self):
        return 'Signature({:x},{:x})'.format(self.r, self.s)


# private/public key class
class PrivateKey:

    def __init__(self, secret):
        self.secret = secret # private key
        self.point = secret * G  # public key

    def hex(self):
        return '{:x}'.format(self.secret).zfill(64)

    # the signing algorithm
    def sign1(self, z):
        # This is for you to implement
        # You should return an object of class Signature(r,s)

        # I'll give you some examples
        # privKey = PrivateKey(256) ## This ought to be secure :-)
        # z = 25
        # What is the signature???
        r=s=1
        return Signature(r, s)

# signature verification algorithm
def verify(pubKey, z, sig):
    # This is for you to implement
    # Verify whether your signatures work with the same example above

    return false


'''
pk = PrivateKey(256) 
pubKey = pk.point

z = 25

sig = pk.sign1(z)

print(sig)

# remember 0x in front of the hex representation
signature = Signature(0xace947112d666ca530210113042ac5978f81662adb215aa70c1ce7d42e0b66cf,0x27e992116152fc070dfdc564b39b204462e9f13a38511861ef0770b662483b84)

print(verify(pubKey,z, signature))
'''

'''
print(verify(pubKey,z, sig))
print(sig)
print(pk.point)


# I'm siging "ints" (points on the curve)
# How to sign a real message?
# Easy; just convert the hexdigest of a hash to an int
z = int(hash256(b'Hello!').hex(),16) 


sig = pk.sign1(z)


print(verify(pubKey,z, sig))
print(sig)

'''

