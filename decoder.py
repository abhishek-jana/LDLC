import numpy as np

# Create an integer messege vector b

b = np.array([3,6,7,2,5,2])

# create H here n = 6 d = 3 normalize H to get |det(G)| = |det(H)| = 1

H = np.array([[0, -0.8, 0, -0.5, 1, 0],[0.8, 0, 0, 1, 0, -0.5],[0, 0.5, 1, 0, 0.8, 0],[0, 0, -0.5, -0.8, 0, 1],[1, 0, 0, 0, 0.5, 0.8],[0.5, -1, -0.8, 0, 0, 0]])

H_norm = H/np.abs(np.linalg.det(H))**(1/6.)

# Calculate generator matrix G = inv(H) normalize H to get |det(G)| = 1

G = np.linalg.inv(H_norm)

# Calculate codeword x

x = np.dot(G,b)

# create noisy codeword y = x + w

mu, sigma = 0, 1 # mean and standard deviation
w = np.random.normal(mu, sigma, x.shape)
y = x + w

# x1, x2, x3, x4,...,xn -> variable node c1,c2,....,cn -> check node
# Initialization

x_input = np.arange(-200.0,200.0,0.1)

def messege(k,x):
    # y and sigma should be given
    global y
    global sigma
    return (np.exp(-(((y[k] - x)/sigma)**2)/2.))/(np.sqrt(2*np.pi*sigma**2))

# Basic iteration

def conv(j,h_r,x):
    # assuming r is known
    # h_r is the non-zero elements of a row
    p_j = 1
    for i,k in enumerate (h_r[:j]):
        p_j = np.convolve(p_j,messege(i,x/k))
    for l,m in enumerate (h_r[j:]):
        p_j = np.convolve(p_j,messege(l,x/m))
    return p_j

def stretch(j,h_r,x):
    # The result is stretched by -h_j
    return conv(j,h_r,-h_r[j]*x)

def periodic_extension(j,h_r,x):
    #The result is extended to a periodic function with period 1/|hj |:
    end = 1000
    i = -1000
    q = 0
    while i < end:
        q += stretch(j, h_r , x  - i/h_r[j])
        i += 1
    return q
    
#print (conv(1,np.array([-0.8,-0.5,1])))
#print (messege(1,x))
    
    
#test = stretch(1,[-0.8,-0.5,1],x_input)
#nonzero = [e for e in test if e!= 0]

#print (len(nonzero),len(test))

print (len(periodic_extension(1,[-0.8,-0.5,1],x_input)))

