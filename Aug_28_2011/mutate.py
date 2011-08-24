def pf1(x):
    x = 1
    
def pf2(x):
    if not isinstance(x,list):
        raise ValueError, "passing argument must be a list"
    x[0] = 1

def pf3(x):
    x.a = 100

def npf1(x):
    print x.__array_interface__["data"][0]
    x[0] = 1
    print x.__array_interface__["data"][0]

def npf2(x):
    print x.__array_interface__["data"][0]
    x = x + 1
    print x.__array_interface__["data"][0]
    return x

def npf3(x):
    print x.__array_interface__["data"][0]
    x += 1
    print x.__array_interface__["data"][0]
