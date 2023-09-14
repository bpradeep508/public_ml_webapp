import time
import random
#import pycuda.autoinit
import nufhe
import numpy as np
import numpy as np
from datetime import datetime

def cba4(r,a,b,carry):
    temp= vm.gate_and(a, b) #ab
    temp1= vm.gate_and( b,carry ) #bc
    temp2= vm.gate_and(a, carry) # ac
    temp3= vm.gate_or(temp, temp1) #ab+ac
    r[1]=vm.gate_or(temp3,temp2) # ab+abc+ac
    r[0]=vm.gate_not(r[1]) #(ab+bc+ca)'
    return r
def cba1(r,a,b,carry):
    temp= vm.gate_and(b,carry) #temp=bc
    r[1]=vm.gate_or(temp, a)   #r1=bc+a 
    r[0]=vm.gate_not(r[1])     #r0=(bc+a)'
    return r

def threetotwo(r,p0,p1,p2):
    r[0] = vm.gate_or(p0, p1)
    temp=vm.gate_and(p0, p1)
    r[1]=vm.gate_or(temp, p2)
    return r
def twotoone(r, p0,p1 ):
    r[0] = vm.gate_or(p0, p1)
    return r

def subtractBits(r, a, b, carry):
    # Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)      #axorb
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry) #axorbxorc
    # And(t1[0], t1[0], t2[0])
    acomp=vm.gate_not(a)   #a'
    abcomp=vm.gate_not(t1)  #axorb'
    t2 = vm.gate_and(acomp, b)
    t3 = vm.gate_and(abcomp, carry)
    #t4=vm.gate_and(t2,t3)
    #t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])
    r[1] = vm.gate_or(t2, t3)

    return r
def subtractNumbers(ctA, ctB, nBits):
    ctRes = [[vm.empty_ciphertext((1,))] for i in range(0,nBits)]
    # carry = vm.empty_ciphertext((1,))
    bitResult = [[vm.empty_ciphertext((1,))] for i in range(2)]
    ctRes[0] = vm.gate_xor(ctA[0], ctB[0])
    # Xor(ctRes[0], ctA[0], ctB[0])
    t1=vm.gate_not(ctA[0])
    carry = vm.gate_and(t1, ctB[0])
    # And(carry[0], ctA[0], ctB[0])
    for i in range(1,nBits ):
        if i>16:
          #bitResult = twotoone(bitResult, ctA[i], ctB[i])
          bitResult = threetotwo(bitResult, ctA[i], ctB[i],carry)
          #Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])
          carry = nufhe.LweSampleArray.copy(bitResult[1])
        else:
        
          bitResult = subtractBits(bitResult, ctA[i], ctB[i], carry)
          # Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

          # Copy(carry[0], bitResult[1])
          carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
#def bubblesort(distances)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row,nbits)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors




def euclidean_distance(row1, row2,nBits):
    ones= [[vm.empty_ciphertext((1,))] for i in range(nBits)]
    distance=  [ctx.encrypt(secret_key, [False]) for _ in range(nBits)]
    c_zero= [[vm.empty_ciphertext((1,))] for i in range(nBits)]
    #distance = 0.0
    '''for i in range(nBits):
        temp=vm.gate_copy(row2[i])
        ones[i]=vm.gate_not(temp)
    neg2=make_neg(ones,nBits)'''
    
    
    result=subtractNumbers(row1,row2,nBits)
    signbit=vm.gate_copy(result[len(result)-1])
    for i in range(nBits):
        temp=vm.gate_copy(result[i])
        ones[i]=vm.gate_not(temp)
    neg2=make_neg(ones,nBits)      
    for i in range(0,nBits):
           distance[i]=vm.gate_mux(signbit,neg2[i],result[i])
    return distance



def make_neg(n,nbits):
    list1=[int(i) for i in range(0,len(n)) ]
    listp=[]
    #one=vm.empty_ciphertext((1,))
    #zero=[vm.empty_ciphertext((1,))]
    #one= [ctx.encrypt(secret_key, [True])]
    #zero= [ctx.encrypt(secret_key, [False])]
    #t2= [ctx.encrypt(secret_key, [False])]
    #one=  [ctx.encrypt(secret_key, [False]) for _ in range(nbits)]
    #zero=  [ctx.encrypt(secret_key, [False) for _ in range(nbits)]
    #print("type of one is",type(one))
    #print("typeof zero is",type(zero))
    #print("tupe of one is",one)
    #[vm.empty_ciphertext((1,))]
    '''for i in range(0,len(n)):
        #temp=vm.gate_copy(n[i])
        temp=vm.gate_mux(n[i],zero[0],one[0])
        n[i]=temp[:]'''
            
    
    #print(n)
    one= ctx.encrypt(secret_key, [True])
    onep=  [ctx.encrypt(secret_key, [False]) for _ in range(nbits)]
    onep[0]=one 
    testone= [[vm.empty_ciphertext((1,))] for i in range(nbits)]
    testone=onep[:]
    '''temp1=n[:]
    temp1.reverse()
    result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(nBits)]
    pa=listToString(result_bits)
    print("n before",twos_comp_val(int(pa,2),len(pa)))'''
    n=addNumbers(n,testone,nbits)
    '''temp1=n[:]
    temp1.reverse()
    result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(nBits)]
    pa=listToString(result_bits)
    print("n after",twos_comp_val(int(pa,2),len(pa)))'''
    return n


def fixSizeBoolList(decimal,size):
    x = [int(x) for x in bin(decimal)[2:]]
    x = list(map(bool, x))
    x = [False]*(size - len(x)) + x
    pow2 = []
    for i in range(size):
        pow2.append([x[i]])
    pow2.reverse()
    return pow2
def boolListToInt(bitlists):
    out = 0
    for bit in bitlists:
        out = (out << 1) | bit
    return out

def addBits(r, a, b, carry):
    # Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])
    r[1] = vm.gate_or(t5, t4)

    return r
def addNumbers(ctA, ctB, nBits):
    ctRes = [[vm.empty_ciphertext((1,))] for i in range(nBits)]
    # carry = vm.empty_ciphertext((1,))
    bitResult = [[vm.empty_ciphertext((1,))] for i in range(2)]
    ctRes[0] = vm.gate_xor(ctA[0], ctB[0])
    # Xor(ctRes[0], ctA[0], ctB[0])
    carry = vm.gate_and(ctA[0], ctB[0])
    # And(carry[0], ctA[0], ctB[0])
    for i in range(1,nBits ):
        if i<0:
          #bitResult = twotoone(bitResult, ctA[i], ctB[i])
          bitResult = threetotwo(bitResult, ctA[i], ctB[i],carry)
          #Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])
          carry = nufhe.LweSampleArray.copy(bitResult[1])

        else:
          bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
          # Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])
          # Copy(carry[0], bitResult[1])
          carry = nufhe.LweSampleArray.copy(bitResult[1])
            
        
    return ctRes
def Convert_list(string):
    list1=[]
    list1[:0]=string
    #print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append([False])
        else:
            listb.append([True])    
    
    #print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}" 
    #print(type(a))
    a=Convert_list(a)
    a.reverse()
    #print(a)
    return a
def listToString(s): 
    # initialize an empty string
    list1=[int(i)for i in s ]
    listp=[]
    for i in list1:
        if i==False:
            listp.append('0')
        else:
            listp.append('1')    
    
    #print(listp)
    str1 = "" 
    # traverse in the string  
    s=['delim'.join([str(elem) for elem in sublist]) for sublist in listp]
    #print(s)
    for ele in s: 
        str1 += ele  
    # return string 
    #print(str1)
    return str1
def twos_comp_val(val,bits):
    """compute the 2's complement of int value val"""
    #val=listToString(val)
    
    #print("2's complente of",val,"is")
    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

def compare_bit(  a, b,  lsb_carry,  tmp): 
    result= ctx.encrypt(secret_key, [False])
    tmp=vm.gate_xnor(a, b)
    result=vm.gate_mux(tmp,lsb_carry, a)
    return result


def minimum(  a,  b,  nb_bits): 
    tmps1= ctx.encrypt(secret_key, [False])
    tmps2= ctx.encrypt(secret_key, [True])
    #initialize the carry to 0
    #run the elementary comparator gate n times
    for i in range(nb_bits):
        tmps1= compare_bit(a[i],b[i],tmps1,tmps2)
    #tmps[0] is the result of the comparaison: 0 if a is larger, 1 if b is larger
    #select the max and copy it to the result
    return tmps1

    
        
def predict(ctA,secret_key, output_bits):
    zero = [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
    onen=  [ctx.encrypt(secret_key, [True]) for _ in range(output_bits)]        
    onep=  [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
    one= ctx.encrypt(secret_key, [True])
    temp= ctx.encrypt(secret_key, [True])
    temp=ctA[output_bits-1]
    comp_res= ctx.encrypt(secret_key, [True])
    onep[0]=one
    ctRes = [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
    # Copy(ctRes[i], bitResult[0]);
    #comp_res= minimum(ctA,zero,output_bits)
    #temp=
    #comp_res=
    
    
    
    for i in range(output_bits):
        ctRes[i] = vm.gate_mux(temp,onen[i],onep[i])  
    # Copy(carry[0], bitResult[1])
    return ctRes


def dist_calc(line,line1,nbits):
    #print(line)
    print(type(list))
    
    return wx,b_l













ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
def intialization(train_size,test_size,neigh,lines,lines1):    
    
    size=16
    onep=  [ctx.encrypt(secret_key, [False]) for _ in range(size)]
    onen=  [ctx.encrypt(secret_key, [True]) for _ in range(size)]
    temp_dist=  [ctx.encrypt(secret_key, [False]) for _ in range(size)]
    temp_label=  [ctx.encrypt(secret_key, [False]) for _ in range(size)]
    one= ctx.encrypt(secret_key, [True])
    zero=ctx.encrypt(secret_key, [False])
    onep[0]=vm.gate_copy(one)
    '''with open('/content/drive/MyDrive/train.txt') as f:
        lines = []
        for line in f:
            lines.append([int(v) for v in line.split()])
    with open('/content/drive/MyDrive/test.txt') as f:
        lines1 = []
        for line in f:
            lines1.append([int(v) for v in line.split()])'''
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]
    train_size= int(train_size)
    print("training size",train_size)
    test_size=int(test_size)
    print("test size",test_size)
    neigh=int(neigh)
    print("neighbours",neigh)

    '''b_x0=[]
    b_x1=[]
    b_x2=[]
    b_x3=[]
    for i in range(test_size):
        temp=int(lines[i][0])
        #print(type(temp))
        b_x0.append(fixSizeBoolList(temp,size))
        temp=int(lines[i][1])
        b_x1.append(fixSizeBoolList(temp,size))
        temp=int(lines[i][2])
        b_x2.append(fixSizeBoolList(temp,size))
        temp=int(lines[i][3])
        b_x3.append(fixSizeBoolList(temp,size))'''
    #print(b_x0,b_x1,b_x2,b_x3)
    #-3.5133618, -4.2616279 , 10.59763759,  6.17913037, -0.1758181
    #w_p=[-15,-18 ,-78 ,138 , 62] # scratch x/1+|x|
    w_p=[-1,-3,4,2,0] # sk learn x/1+|x|
    #w_p=[4,-8 ,22 ,9 , -1] sigmoid
    #w_p=[-3, -4 , 10,  6, -1] svm
    
    #print(type(x[0]))
    #print(x)
    #b_y = fixSizeBoolList(deci_y,size)
    #print(type(y[0]))
    #print(y)
    #x.reverse()
    #print(x)
    #y.reverse()
    #print(y)
    featuresize=5
    '''w=[]
    w_b=[]
    for i in range(featuresize-1):
        w.append([[vm.empty_ciphertext((1,))] for i in range(size)])
        w_b.append(twos_complement(w_p[i],size))
        for j in range(size):
            w[i][j] = ctx.encrypt(secret_key, w_b[i][j])
    bias=[[vm.empty_ciphertext((1,))] for i in range(size)]
    bias_b=twos_complement(w_p[4],size)'''
    '''eten=[[vm.empty_ciphertext((1,))] for i in range(size)]
    eten_b=twos_complement(10,size)
    for i in range(size):
        bias[i]=ctx.encrypt(secret_key, bias_b[i])
        eten[i]=ctx.encrypt(secret_key, eten_b[i])
    '''
    #train_data = pd.read_csv('/home/pradeep/Desktop/train.txt',sep=" ",header = None)
    #test_data=pd.read_csv('/home/pradeep/Desktop/test.txt',sep=" ",header = None)
    dist=[]
    for i in range(train_size):
        dist.append([[ctx.encrypt(secret_key, [False]) for i in range(16)] for j in range(2)]) 
    
    plain_predict=[]
    start_time = datetime.now()
    print("start time",start_time)
    #dist=[[vm.empty_ciphertext((1,))] for i in range(train_size) for j in range(size)]
    ciphertext1a=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext2a=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext3a=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext4a=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext1=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext2=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext3=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertext4=[[vm.empty_ciphertext((1,))] for i in range(size)]
    ciphertextl=[[vm.empty_ciphertext((1,))] for i in range(size)]
    for j in range(test_size):
        print("j values is",j)
        print("################################")
        
        temp1=int(lines1[j][0])
        b_y0=twos_complement(temp1,size)
        print(temp1,"2's complement is",b_y0)
        temp1=int(lines1[j][1])
        b_y1=twos_complement(temp1,size)
        print(temp1,"2's complement is",b_y1)
        temp1=int(lines1[j][2])
        b_y2=twos_complement(temp1,size)
        print(temp1,"2's complement is",b_y2)
        temp1=int(lines1[j][3])
        b_y3=twos_complement(temp1,size)
        print(temp1,"2's complement is",b_y3)
        for k in range(size):
          ciphertext1a[k] = ctx.encrypt(secret_key, b_y0[k])
          ciphertext2a[k] = ctx.encrypt(secret_key, b_y1[k])
          ciphertext3a[k] = ctx.encrypt(secret_key, b_y2[k])
          ciphertext4a[k] = ctx.encrypt(secret_key, b_y3[k])
        for i in range(train_size):
            print("i value is ",i)
            print(lines[i][0],lines[i][1],lines[i][2],lines[i][3],lines[i][4])
            print(lines1[j][0],lines1[j][1],lines1[j][2],lines1[j][3])
            temp=int(lines[i][0])
            b_x0=twos_complement(temp,size)
            print(temp,"2's complement is",b_x0)
            temp=int(lines[i][1])
            b_x1=twos_complement(temp,size)
            print(temp,"2's complement is",b_x1)
            temp=int(lines[i][2])
            b_x2=twos_complement(temp,size)
            print(temp,"2's complement is",b_x2)
            temp=int(lines[i][3])
            b_x3=twos_complement(temp,size)
            print(temp,"2's complement is",b_x3)
            temp=int(lines[i][4])
            b_l=twos_complement(temp,size)
            
            
            
            for _ in range(size):
              ciphertext1[_] = ctx.encrypt(secret_key, b_x0[_])
              ciphertext2[_] = ctx.encrypt(secret_key, b_x1[_])
              ciphertext3[_] = ctx.encrypt(secret_key, b_x2[_])
              ciphertext4[_] = ctx.encrypt(secret_key, b_x3[_])
              ciphertextl[_] = ctx.encrypt(secret_key, b_l[_])        
            #ciphertext1 = ctx.encrypt(secret_key, x)
            #ciphertext2 = ctx.encrypt(secret_key, y) 
            start_time = time.time()
            #ciphertext1.reverse()
            #ciphertext2.reverse()
            #result = addNumbers(ciphertext1, ciphertext2, size)
            #partial_predict=[]

            #print(type(ciphertext1[0][0]))
            temp1 = [[vm.empty_ciphertext((1,))] for i in range(size)]
            temp2 = [[vm.empty_ciphertext((1,))] for i in range(size)]
            #wx=[[vm.empty_ciphertext((1,))] for i in range(size)]
            #predict[i]=[[vm.empty_ciphertext((1,))] for i in range(size*2)]
            #temp1=[]
            #temp2=[]
            print("==============================================")
            print("Distance computation between ",lines[i][0],lines[i][1],lines[i][2],lines[i][3],lines[i][4],"and",lines1[j][0],lines1[j][1],lines1[j][2],lines1[j][3])
            presult_mul1 = euclidean_distance(ciphertext1,ciphertext1a, size)
            #print("ciphertext one",type(ciphertext1))
            #print("ciphertext one inside",type(ciphertext1[0]))
            temp1=presult_mul1[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("feature 1 distance",twos_comp_val(int(pa,2),len(pa)))
            presult_mul2 = euclidean_distance(ciphertext2,ciphertext2a , size)
            temp1=presult_mul2[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("feature 2 distance",twos_comp_val(int(pa,2),len(pa)))
            presult_mul3 = euclidean_distance(ciphertext3,ciphertext3a,  size )
            temp1=presult_mul3[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("feature 3 distance",twos_comp_val(int(pa,2),len(pa)))
            presult_mul4 = euclidean_distance(ciphertext4, ciphertext4a,  size)
            temp1=presult_mul4[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("feature 4 distance",twos_comp_val(int(pa,2),len(pa)))
            presult_add1 = addNumbers(presult_mul1, presult_mul2,  size)
            '''temp1=presult_add1[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("firstadd",pa)
            print("add1",twos_comp_val(int(pa,2),len(pa)))'''
            presult_add2 = addNumbers(presult_mul3, presult_mul4,  size)
            '''temp1=presult_add2[:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            #print("secondadd",pa)
            print("add2",twos_comp_val(int(pa,2),len(pa)))'''
            dist[i][0]= addNumbers(presult_add1, presult_add2,  size)
            temp1=dist[i][0][:]
            temp1.reverse()
            result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
            pa=listToString(result_bits)
            print("secondadd",pa)
            print("distance is",twos_comp_val(int(pa,2),len(pa)))
            #print(type(temp))
            #x = twos_complement(deci_x,size)
            for k in range(size):
              dist[i][1][k]=vm.gate_copy(ciphertextl[k])   
        print("====================================================")
        print("sorting of distances")
        for l in range( train_size):    
            # Last i elements are already in place
            for m in range(train_size-l-1):
                #if (arr[j] > arr[j+1]):
                #swap(&arr[j], &arr[j+1]);
                temp=subtractNumbers(dist[m][0],dist[m+1][0], size)
                '''temp1=temp[:]
                temp1.reverse()
                result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
                pa=listToString(result_bits)
                print("temp",twos_comp_val(int(pa,2),len(pa)))'''
                
                signbit= ctx.encrypt(secret_key, [False])
                signbitnot= ctx.encrypt(secret_key, [False])
                signbit=temp[size-1]
                
                signbitnot=vm.gate_not(signbit)
                tempb1=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempb2=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempb3=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempb4=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempn1=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempn2=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempn3=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                tempn4=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
                for _ in range(size):
                    tempb1[_]=dist[m][0][_]
                    tempb2[_]=dist[m+1][0][_]
                    tempb3[_]=dist[m][1][_]
                    tempb4[_]=dist[m+1][1][_]
                '''temp1=tempb1[:]
                temp1.reverse()
                result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
                pa=listToString(result_bits)
                print("tempb1",twos_comp_val(int(pa,2),len(pa)))
                temp1=tempb2[:]
                temp1.reverse()
                result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
                pa=listToString(result_bits)
                print("tempb2",twos_comp_val(int(pa,2),len(pa)))
                temp1=tempb3[:]
                temp1.reverse()
                result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
                pa=listToString(result_bits)
                #print("tempb3",pa)
                print(" tempb3",twos_comp_val(int(pa,2),len(pa)))
                temp1=tempb4[:]
                temp1.reverse()
                result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
                pa=listToString(result_bits)
                #print("wx is",pa)
                print(" tempb4",twos_comp_val(int(pa,2),len(pa)))'''        
                
                for k in range(size):
                    dist[m][0][k]=vm.gate_mux(signbit,dist[m][0][k],dist[m+1][0][k])
                    dist[m+1][0][k] =vm.gate_mux(signbit,tempb2[k],tempb1[k])
                    dist[m][1][k]=vm.gate_mux(signbit,dist[m][1][k],dist[m+1][1][k])
                    dist[m+1][1][k]=vm.gate_mux(signbit,tempb3[k],tempb4[k]) 
                    
        pos=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        neg=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        #predict1=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        tempp=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        posc=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        negc=[ctx.encrypt(secret_key, [False]) for _ in range(size)]
        signbitl= ctx.encrypt(secret_key, [False])
        signbitc= ctx.encrypt(secret_key, [False])
        
        print("neighbours voting")
        for i in range(neigh):
            signbitl=vm.gate_copy(dist[i][1][size-1])
            signplain=[]
            signplain.append([ctx.decrypt(secret_key,signbitl)])
            print("signbit of label ",signplain)
            signbitc=vm.gate_copy(onep[size-1])
            signplain=[]
            signplain.append([ctx.decrypt(secret_key,signbitc)])
            print("signbit of 1",signplain)
            dbit=vm.gate_xnor(signbitl,signbitc)

            posc=pos[:]
            negc=neg[:]
            pos=make_neg(pos,size)
            neg=make_neg(neg,size)
            for j in range(size):
                pos[j]=vm.gate_mux(dbit,pos[j],posc[j])
                neg[j]=vm.gate_mux(dbit,negc[j],neg[j])     
        temp1=pos[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("poscount",twos_comp_val(int(pa,2),len(pa)))
        temp1=neg[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("neg count",twos_comp_val(int(pa,2),len(pa)))
       
        
        temp=subtractNumbers(pos,neg,size)
        temp1=temp[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("pos-neg",twos_comp_val(int(pa,2),len(pa)))
        signbit= vm.gate_copy(temp[size-1])
        for i in range(size):
            tempp[i]=vm.gate_mux(signbit,onen[i],onep[i])
        predict1=tempp[:]
        predict1.reverse()
        print(predict1)
        #print(type(predict1))
        #print(type(predict1[0]))
        #result_bits = [ctx.decrypt(secret_key, predict1[i]) for i in range(size*2)]
        tempp.reverse()
        result_bits = [ctx.decrypt(secret_key, tempp[i]) for i in range(size)]
        #plain_predict.append(boolListToInt(result_bits))
        pa=listToString(result_bits)
        print("cloud side encrypted value result")
        print(pa)
        print("===================================")
        print("plaind predicted value after client side decryption")
        print("preddicted",twos_comp_val(int(pa,2),len(pa)))
        plain_predict.append(twos_comp_val(int(pa,2),len(pa)))
        print("j values is",j,"endtime",datetime.now(),"start time",start_time)   
    #print("start time is",start_time)
    #print("end time",time.time())
    print("endtime",datetime.now(),"start time",start_time)
    #result.reverse()
    for j in range(test_size):
        print(" nuFHE multiplication number is actual",lines[j][4],"preicted", plain_predict[j])
    return plain_predict