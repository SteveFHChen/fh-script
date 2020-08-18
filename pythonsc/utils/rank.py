
def rank(list):
    rst = []
    i=1; j=1
    p=list[0]
    for s in list:
        if(len(rst)>=1):
            if (p!=s):
                i = j
                p=s
        rst.append(i)
        j+=1
    return rst

def dense_rank(list):
    rst = []
    i=1
    p=list[0]
    for s in list:
        if(len(rst)>=1):
            if (p!=s):
                i = i+1
                p=s
        rst.append(i)
    return rst

def distinct(list):
    rst = []
    i=1
    rst.append(list[0])
    p=list[0]
    for s in list:
        if(len(rst)>=1):
            if (p!=s):
                rst.append(s)
                p=s
    return rst

#Unit test:
'''
a1=[[20200101],[20200102],[20200102],[20200103],[20200104],[20200104],[20200105]]

print("Exprected result: [0, 1, 1, 3, 4, 4, 6]")
print("Real result: ", rank(a1))

print("Exprected result: [0, 1, 1, 2, 3, 3, 4]")
print("Real result: ",dense_rank(a1))

print("Distinct result:", distinct(a1))
'''
