# -*- coding: utf-8 -*-





import time
import random
import nufhe
import numpy as np
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
def approaddBits(r, a, b, carry):
    '''# Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])po
    r[1] = vm.gate_or(t5, t4)'''
    '''r[1]= vm.gate_or(a, b)    #p0 + p1
    temp=vm.gate_and(a, b)
    r[0]=vm.gate_or(carry, temp)   #p0 p1 + p2
    #r[1]=vm.gate_or(w0, w1)
    #r[1] = vm.gate_and(w0, w1)'''


    #cba1
    temp= vm.gate_and(b,carry)
    r[1]=vm.gate_or(temp, a)
    r[0]=vm.gate_not(r[1])
    '''#cba2
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_xor(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba3
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_or(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba4
    temp= vm.gate_and(a, b)
    temp1= vm.gate_and( b,carry)
    temp2= vm.gate_and(a, carry)
    temp3= vm.gate_or(temp, temp1)
    r[1]=vm.gate_or(temp3,temp2)
    r[0]=vm.gate_not(r[1])'''
    '''# Xor(t1[0], a, carry[0])
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
    r[1] = vm.gate_or(t5, t4)'''

    return r

def addBits(r, a, b, carry):
    '''# Xor(t1[0], a, carry[0])
    t1 = vm.gate_xor(a, b)
    # Xor(t2[0], b, carry[0])
    # Xor(r[0], a, t2[0])
    r[0] = vm.gate_xor(t1, carry)
    # And(t1[0], t1[0], t2[0])
    t2 = vm.gate_and(a, carry)
    t3 = vm.gate_and(b, carry)
    t4=vm.gate_and(a,b)
    t5= vm.gate_or(t2,t3)
    # Xor(r[1], carry[0], t1[0])po
    r[1] = vm.gate_or(t5, t4)'''
    '''r[1]= vm.gate_or(a, b)    #p0 + p1
    temp=vm.gate_and(a, b)
    r[0]=vm.gate_or(carry, temp)   #p0 p1 + p2
    #r[1]=vm.gate_or(w0, w1)
    #r[1] = vm.gate_and(w0, w1)'''


    '''#cba1
    temp= vm.gate_and(b,carry)
    r[1]=vm.gate_or(temp, a)
    r[0]=vm.gate_not(r[1])'''
    '''#cba2
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_xor(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba3
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_or(temp, carry)
    r[0]=vm.gate_not(r[1])
    #cba4
    temp= vm.gate_and(a, b)
    temp1= vm.gate_and( b,carry)
    temp2= vm.gate_and(a, carry)
    temp3= vm.gate_or(temp, temp1)
    r[1]=vm.gate_or(temp3,temp2)
    r[0]=vm.gate_not(r[1])'''
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
        if i<4:
          bitResult = approaddBits(bitResult, ctA[i], ctB[i], carry)
          # Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])
        # Copy(carry[0], bitResult[1])
          carry = nufhe.LweSampleArray.copy(bitResult[1])
        else:
          bitResult =addBits(bitResult, ctA[i], ctB[i], carry)
          # Copy(ctRes[i], bitResult[0]);
          ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])
        # Copy(carry[0], bitResult[1])
          carry = nufhe.LweSampleArray.copy(bitResult[1])
    return ctRes


def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    temp = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [ctx.encrypt(secret_key, [False]) for _ in
                      range(output_bits)]
        #temp=mux(temp,ctA,ctB[i],size)
        for j in range(input_bits):
            andResLeft[j + i] = vm.gate_and(ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        #result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]
        #print(" nuFHE multiplication intermdiate number is : ",i,boolListToInt(result_bits))

    return result[:size]
def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
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


    
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
    
def intialization(lines,test_size=1):    
    size=16
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]
    lines = []
    for line in f:
        lines.append([int(v) for v in line.split()])
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
    # 0.18989317 -0.25418705  0.02612883  0.00184397 -0.12274228 -0.01697259 -0.01228785 -0.06611546 -0.08922483  0.16153068 -0.02537609
    #w_p=[-3, -4 , 10,  6, -1]
    w_p=[-4, 6, -1, -1, 3, 1, 1, 2, 2, -4, 1, 6]
    #print(type(x[0]))
    #print(x)
    #b_y = fixSizeBoolList(deci_y,size)
    #print(type(y[0]))
    #print(y)
    #x.reverse()
    #print(x)
    #y.reverse()
    #print(y)
    featuresize=12
    w=[]
    w_b=[]
    for i in range(featuresize-1):
        w.append([[vm.empty_ciphertext((1,))] for i in range(size)])
        w_b.append(twos_complement(w_p[i],size))
        for j in range(size):
            w[i][j] = ctx.encrypt(secret_key, w_b[i][j])
    bias=[[vm.empty_ciphertext((1,))] for i in range(size)]
    bias_b=twos_complement(w_p[len(w_p)-1],size)
    for i in range(size):
      bias[i]=ctx.encrypt(secret_key, bias_b[i])

    plain_predict=[]
    start_time = time.time()
    for i in range(test_size):
        print(lines[i][0],lines[i][1],lines[i][2],lines[i][3])
        temp=int(lines[i][0])
        b_x0=twos_complement(temp,size)
        temp=int(lines[i][1])
        b_x1=twos_complement(temp,size)
        temp=int(lines[i][2])
        b_x2=twos_complement(temp,size)
        temp=int(lines[i][3])
        b_x3=twos_complement(temp,size)
        temp=int(lines[i][4])
        b_x4=twos_complement(temp,size)
        temp=int(lines[i][5])
        b_x5=twos_complement(temp,size)
        temp=int(lines[i][6])
        b_x6=twos_complement(temp,size)
        temp=int(lines[i][7])
        b_x7=twos_complement(temp,size)
        temp=int(lines[i][8])
        b_x8=twos_complement(temp,size)
        temp=int(lines[i][9])
        b_x9=twos_complement(temp,size)
        temp=int(lines[i][10])
        b_x10=twos_complement(temp,size)



        ciphertext1=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext2=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext3=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext4=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext5=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext6=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext7=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext8=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext9=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext10=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext11=[[vm.empty_ciphertext((1,))] for i in range(size)]


        for j in range(size):
            ciphertext1[j] = ctx.encrypt(secret_key, b_x0[j])
            ciphertext2[j] = ctx.encrypt(secret_key, b_x1[j])
            ciphertext3[j] = ctx.encrypt(secret_key, b_x2[j])
            ciphertext4[j] = ctx.encrypt(secret_key, b_x3[j])
            ciphertext5[j] = ctx.encrypt(secret_key, b_x4[j])
            ciphertext6[j] = ctx.encrypt(secret_key, b_x5[j])
            ciphertext7[j] = ctx.encrypt(secret_key, b_x6[j])
            ciphertext8[j] = ctx.encrypt(secret_key, b_x7[j])
            ciphertext9[j] = ctx.encrypt(secret_key, b_x8[j])
            ciphertext10[j] = ctx.encrypt(secret_key, b_x9[j])
            ciphertext11[j] = ctx.encrypt(secret_key, b_x10[j])

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
        #predict[i]=[[vm.empty_ciphertext((1,))] for i in range(size*2)]
        #temp1=[]
        #temp2=[]
        presult_mul1 = mulNumbers(ciphertext1,w[0], secret_key, size, size * 2)
        '''temp1=presult_mul1[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul1",twos_comp_val(int(pa,2),len(pa)))'''
        presult_mul2 = mulNumbers(ciphertext2, w[1], secret_key, size, size * 2)
        '''temp1=presult_mul2[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul2",twos_comp_val(int(pa,2),len(pa)))'''
        presult_mul3 = mulNumbers(ciphertext3, w[2], secret_key, size, size * 2)
        '''temp1=presult_mul3[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul3",twos_comp_val(int(pa,2),len(pa)))'''
        presult_mul4 = mulNumbers(ciphertext4, w[3], secret_key, size, size * 2)
        '''temp1=presult_mul4[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("mul4",twos_comp_val(int(pa,2),len(pa)))'''
        presult_mul5 = mulNumbers(ciphertext5, w[4], secret_key, size, size * 2)
        presult_mul6 = mulNumbers(ciphertext6, w[5], secret_key, size, size * 2)
        presult_mul7 = mulNumbers(ciphertext7, w[6], secret_key, size, size * 2)
        presult_mul8 = mulNumbers(ciphertext8, w[7], secret_key, size, size * 2)
        presult_mul9 = mulNumbers(ciphertext9, w[8], secret_key, size, size * 2)
        presult_mul10 = mulNumbers(ciphertext10, w[9], secret_key, size, size * 2)
        presult_mul11 = mulNumbers(ciphertext11, w[10], secret_key, size, size * 2)


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
        print("secondadd",pa)
        print("add2",twos_comp_val(int(pa,2),len(pa)))'''
        presult_add3 = addNumbers(presult_mul5, presult_mul6,  size)
        presult_add4 = addNumbers(presult_mul7, presult_mul8,  size)
        presult_add5 = addNumbers(presult_mul9, presult_mul10,  size)
        #first step
        presult_add11 = addNumbers(presult_add1, presult_add2,  size)
        presult_add12 = addNumbers(presult_add3, presult_add4,  size)
        presult_add13 = addNumbers(presult_add5, presult_mul11,  size)
        #secong step
        presult_add21 = addNumbers(presult_add11, presult_add12,  size)
        presult_add22 = addNumbers(presult_add13, bias,  size)
        '''temp1=presult_add3[:]
        temp1.reverse()
        result_bits = [ctx.decrypt(secret_key, temp1[i]) for i in range(size)]
        pa=listToString(result_bits)
        print("thirdadd",pa)
        print("add3",twos_comp_val(int(pa,2),len(pa)))'''


        partial_predict = addNumbers(presult_add21,presult_add22,  size)
        #partial_predict.revesrse()
        result_bits = [ctx.decrypt(secret_key, partial_predict[i]) for i in range(size)]
        result_bits.reverse()
        pa=listToString(result_bits)
        print("partial predict",twos_comp_val(int(pa,2),len(pa)))
        predict1=predict(partial_predict,secret_key,size)
        predict1.reverse()
        #print(predict1)
        #print(type(predict1))
        #print(type(predict1[0]))
        #result_bits = [ctx.decrypt(secret_key, predict1[i]) for i in range(size*2)]
        result_bits = [ctx.decrypt(secret_key, predict1[i]) for i in range(size)]
        #plain_predict.append(boolListToInt(result_bits))

        pa=listToString(result_bits)
        plain_predict.append(twos_comp_val(int(pa,2),len(pa)))
        print(" nuFHE multiplication number is : ", plain_predict[i])
    print("multiplication time",time.time() - start_time)
    #result.reverse()
    pos=0
    neg=0
    ''' for i in range(test_size):
      if int(lines[i][4])==int(plain_predict[j]):
        pos=pos+1
      else:
        neg=neg+1
    print("acc is =",pos/test_size)'''
    for j in range(test_size):
        print(" nuFHE multiplication number is actual",lines[j][4],"preicted", plain_predict[j])

import time
import random
import nufhe
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
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes

'''def mux(ca,cb,s,size):
    #size=len(ca)
    result = [ctx.encrypt(secret_key, [False]) for _ in range(size*2)]
    result=[]
    for i in range(size):
        result.append(vm.gate_mux(s,cb[i],ca))
    return result
'''

def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    temp = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [ctx.encrypt(secret_key, [False]) for _ in
                      range(output_bits)]
        #temp=mux(temp,ctA,ctB[i],size)
        for j in range(input_bits):
            andResLeft[j + i] = vm.gate_and(ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        #result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]
        #print(" nuFHE multiplication intermdiate number is : ",i,boolListToInt(result_bits))

    return result[:size]














def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append([False])
        else:
            listb.append([True])

    print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
    print(type(a))
    a=Convert_list(a)
    a.reverse()
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

    print(listp)
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


    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

begin = time.time()
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
# size even decimal only
size = 64
# test decimal number. bits lenght need to be less than size/2
deci_x = 10
deci_y = 1

x = twos_complement(deci_x,size)
print(type(x))
print(x)
y = twos_complement(deci_y,size)
print(type(y))
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

carry = vm.empty_ciphertext((1,))
for i in range(16):
  CiX[i]=vm.gate_not(CiX[i])
res=addNumbers(ciX,ciY,size)
res_mul=mulNumbers(ciX,ciY, secret_key, size, size * 2)
#print(type(res))
#print(res)
#print(type(res[0]))
y=[ctx.decrypt(secret_key, res[i]) for i in range(size)]
y_mul=[ctx.decrypt(secret_key, res_mul[i]) for i in range(size)]
y_mul.reverse()
y.reverse()
pa=listToString(y)
pb=listToString(y_mul)
#print(pa)
#pa=listToString(pa)
print("pa is ",pa)
print("pb is ",pb)
inta=twos_comp_val(int(pa,2),len(pa))
intb=twos_comp_val(int(pb,2),len(pb))
print(inta)
print(intb)
#intb=twos_comp_val(int(pb,2),len(pb))


#print("Adder result:",twos_comp_val(int(x,2),len(x)),twos_comp_val(int(y,2),len(y)),twos_comp_val(int("".join(map(str, y)),2),len(y)))

# store end time
end = time.time()

# total time taken
print(f"Total runtime of the program is {int(end - begin)} second")

w_p=[0.18989317, -0.25418705,  0.02612883,  0.00184397, -0.12274228, -0.01697259, -0.01228785, -0.06611546, -0.08922483,  0.16153068, -0.02537609,-0.27447494]
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def min_max_norm(dataset):
    if isinstance(dataset, list):
        norm_list = list()
        min_value = min(dataset)
        max_value = max(dataset)

        for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)

    return norm_list
#norm=min_max_norm(w_p)
norm = [float(i)/sum(w_p) for i in w_p]
print(norm)
for i in range(0,len(w_p)):
  norm[i]=norm[i]*10
  if norm[i]<0:
    norm[i]=math.floor(norm[i])
  else:
    norm[i]=math.ceil(norm[i])

print(norm)


#scaler = StandardScaler()
#scaler = MinMaxScaler()
#print(scaler.fit(w_p))
'''for i in range(0,len(w_p)):
  w_p[i]=w_p[i]*100

print(w_p)'''

type(lines[j][4])

import time
import random
import nufhe
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
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes
def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append([False])
        else:
            listb.append([True])

    print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
    print(type(a))
    a=Convert_list(a)
    a.reverse()
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

    print(listp)
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


    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

begin = time.time()
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
# size even decimal only
size = 64
# test decimal number. bits lenght need to be less than size/2
deci_x = 40
deci_y = 50

x = twos_complement(deci_x,size)
print(type(x))
print(x)
y = twos_complement(deci_y,size)
print(type(y))
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

carry = vm.empty_ciphertext((1,))
res=addNumbers(ciX,ciY,size)
#print(type(res))
#print(res)
#print(type(res[0]))
y=[ctx.decrypt(secret_key, res[i]) for i in range(size)]
y.reverse()
pa=listToString(y)
#print(pa)
#pa=listToString(pa)
print(pa)
inta=twos_comp_val(int(pa,2),len(pa))
print(inta)
#intb=twos_comp_val(int(pb,2),len(pb))


#print("Adder result:",twos_comp_val(int(x,2),len(x)),twos_comp_val(int(y,2),len(y)),twos_comp_val(int("".join(map(str, y)),2),len(y)))

# store end time
end = time.time()

# total time taken
print(f"Total runtime of the program is {int(end - begin)} second")

"""approximate adder"""

import time
import random
import nufhe
def addBits(r, a, b, carry):

    w0= vm.gate_or(a, b)    #p0 + p1
    temp=vm.gate_or(a, b)
    w1=vm.gate_or(carry, temp)   #p0 p1 + p2
    r[0]=vm.gate_or(w0, w1)
    r[1] = vm.gate_and(w0, w1)
    '''# Xor(t1[0], a, carry[0])
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
    r[1] = vm.gate_or(t5, t4)'''

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
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes
def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append([False])
        else:
            listb.append([True])

    print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
    print(type(a))
    a=Convert_list(a)
    a.reverse()
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

    print(listp)
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


    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

begin = time.time()
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
# size even decimal only
size = 64
# test decimal number. bits lenght need to be less than size/2
deci_x = 5
deci_y = 20

x = twos_complement(deci_x,size)
print(type(x))
print(x)
y = twos_complement(deci_y,size)
print(type(y))
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

carry = vm.empty_ciphertext((1,))
res=addNumbers(ciX,ciY,size)
#print(type(res))
#print(res)
#print(type(res[0]))
y=[ctx.decrypt(secret_key, res[i]) for i in range(size)]
y.reverse()
pa=listToString(y)
#print(pa)
#pa=listToString(pa)
print(pa)
inta=twos_comp_val(int(pa,2),len(pa))
print(inta)
#intb=twos_comp_val(int(pb,2),len(pb))


#print("Adder result:",twos_comp_val(int(x,2),len(x)),twos_comp_val(int(y,2),len(y)),twos_comp_val(int("".join(map(str, y)),2),len(y)))

# store end time
end = time.time()

# total time taken
print(f"Total runtime of the program is {int(end - begin)} second")

"""

1. approx multiplier

2.   List item

"""

import time
import random
import nufhe
def addBits(r, a, b, carry):
    '''w0= vm.gate_or(a, b)    #p0 + p1
    temp=vm.gate_or(a, b)
    w1=vm.gate_or(carry, temp)   #p0 p1 + p2
    r[0]=vm.gate_or(w0, w1)
    r[1] = vm.gate_and(w0, w1)'''


    '''#cba1
    temp= vm.gate_and(b,carry)
    r[1]=vm.gate_or(temp, a)
    r[0]=vm.gate_not(r[1])'''
    '''#cba2
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_xor(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    '''#cba3
    temp= vm.gate_and(a, b)
    r[1]=vm.gate_or(temp, carry)
    r[0]=vm.gate_not(r[1])'''
    #cba4
    temp= vm.gate_and(a, b)
    temp1= vm.gate_and( b,carry)
    temp2= vm.gate_and(a, carry)
    temp3= vm.gate_or(temp, temp1)
    r[1]=vm.gate_or(temp3,temp2)
    r[0]=vm.gate_not(r[1])
    '''# Xor(t1[0], a, carry[0])
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
    r[1] = vm.gate_or(t5, t4)'''

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
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes

'''def mux(ca,cb,s,size):
    #size=len(ca)
    result = [ctx.encrypt(secret_key, [False]) for _ in range(size*2)]
    result=[]
    for i in range(size):
        result.append(vm.gate_mux(s,cb[i],ca))
    return result
'''

def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    temp = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [ctx.encrypt(secret_key, [False]) for _ in
                      range(output_bits)]
        #temp=mux(temp,ctA,ctB[i],size)
        for j in range(input_bits):
            andResLeft[j + i] = vm.gate_and(ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        #result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]
        #print(" nuFHE multiplication intermdiate number is : ",i,boolListToInt(result_bits))

    return result[:size]














def Convert_list(string):
    list1=[]
    list1[:0]=string
    print(list1)
    list1=[int(i)for i in list1 ]
    listb=[]
    for i in list1:
        if i==0:
            listb.append([False])
        else:
            listb.append([True])

    print(listb)
    return listb

def twos_complement(n,nbits):
    a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
    print(type(a))
    a=Convert_list(a)
    a.reverse()
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

    print(listp)
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


    if (val & (1 << bits - 1)) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

begin = time.time()
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
vm = ctx.make_virtual_machine(cloud_key)
# size even decimal only
size = 16
# test decimal number. bits lenght need to be less than size/2
deci_x = 10
deci_y = -8

x = twos_complement(deci_x,size)
print(type(x))
print(x)
y = twos_complement(deci_y,size)
print(type(y))
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

carry = vm.empty_ciphertext((1,))
res=addNumbers(ciX,ciY,size)
res_mul=mulNumbers(ciX,ciY, secret_key, size, size * 2)
#print(type(res))
#print(res)
#print(type(res[0]))
y=[ctx.decrypt(secret_key, res[i]) for i in range(size)]
y_mul=[ctx.decrypt(secret_key, res_mul[i]) for i in range(size)]
y_mul.reverse()
y.reverse()
pa=listToString(y)
pb=listToString(y_mul)
#print(pa)
#pa=listToString(pa)
print("pa is ",pa)
print("pb is ",pb)
inta=twos_comp_val(int(pa,2),len(pa))
intb=twos_comp_val(int(pb,2),len(pb))
print(inta)
print(intb)
#intb=twos_comp_val(int(pb,2),len(pb))


#print("Adder result:",twos_comp_val(int(x,2),len(x)),twos_comp_val(int(y,2),len(y)),twos_comp_val(int("".join(map(str, y)),2),len(y)))

# store end time
end = time.time()

# total time taken
print(f"Total runtime of the program is {int(end - begin)} second")



def mux(ca,cb,s,size):
    size=len(ca)
    result = [ctx.encrypt(secret_key, [False]) for _ in range(size*2)]
    for i in cb:
        result[i]=vm.gate_mux(s,cb[i],ca[i])
    return result
def multiply(ca,cb,size,out):
    #sign extend
    result = [ctx.encrypt(secret_key, [False]) for _ in range(out)]
    temp = [ctx.encrypt(secret_key, [False]) for _ in range(out)]
    #size = len(ca)

    for i in range(size):
        #temp=np.full_like(np.empty((size, len(ca[0])), dtype=np.uint32), theta_c)
        #temp_res=np.full_like(zero, theta_c)
        temp= mux(temp,ca,cb[i],size)
        for j in range(size):
            temp_res[i+j]=temp[j]
        product=addNumbers(product,temp_res,out)

    return product[:size]

from google.colab import drive
drive.mount('/content/drive')

!cd /content/drive/MyDrive/
!ls

import random
import nufhe
import time

def fixSizeBoolList(decimal,size):
    x = [int(x) for x in bin(decimal)[2:]]
    x = list(map(bool, x))
    x = [False]*(size - len(x)) + x
    return x

# in subtraction, ciX have to be greater than ciY
def subtract(ciX, ciY):
    for i in range(size):
        ciXnotTemp = ciX
        a = vm.gate_and(vm.gate_not(ciX), ciY)
        ciX = vm.gate_xor(ciX, ciY)
        aShiftTemp = a
        aShiftTemp.roll(-1, axis=-1)
        ciY = aShiftTemp

    return ciX

def checkSubtract(sub1,sub2):
    if sub1 > sub2:
        return sub2
    else :
        return sub1

def add(ciX, ciY):
# fixed iteration since
    for i in range(size):
        a = vm.gate_and(ciX, ciY)
        b = vm.gate_xor(ciX, ciY)
        aShiftTemp = a
        # using roll as a shift bit
        aShiftTemp.roll(-1, axis=-1)
        ciX = aShiftTemp
        ciY = b

    return b

def boolListToInt(bitlists):
    out = 0
    for bit in bitlists:
        out = (out << 1) | bit
    return out

### testing ###

ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
# size even decimal only
size = 32
# test decimal number. bits lenght need to be less than size/2
deci_x = 3093
deci_y = 1999

x = fixSizeBoolList(deci_x,size)
print(x)
y = fixSizeBoolList(deci_y,size)
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

# subtraction have to be done twice since we don't know which one is grater than the other
start_sub = time.time()
subXthenY = ctx.decrypt(secret_key, subtract(ciX, ciY))
subYthenX = ctx.decrypt(secret_key, subtract(ciY, ciX))
# the Lesser subtraction result is the right one, this have to be done after decrypting themessage
plainSubtractNumber = checkSubtract(boolListToInt(subXthenY),boolListToInt(subYthenX))
print("subtraction time",time.time() - start_sub)

#print("time taken by subtact is",elap_sub)
print("reference subtract number is : ", (deci_x-deci_y) ,"/ nuFHE subtract number is : ", plainSubtractNumber)



start_add = time.time()
plainAddNumber = ctx.decrypt(secret_key, add(ciX, ciY))


print("addition time",time.time() - start_add)
print("reference add number is : ", (deci_x+deci_y) ,"/ nuFHE subtract number is : ", boolListToInt(plainAddNumber))



import time
import random
import nufhe
import numpy as np
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
        bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        # Copy(ctRes[i], bitResult[0]);
        ctRes[i] = nufhe.LweSampleArray.copy(bitResult[0])

        # Copy(carry[0], bitResult[1])
        carry = nufhe.LweSampleArray.copy(bitResult[1])

    return ctRes


def mulNumbers(ctA, ctB, secret_key, input_bits, output_bits):
    result = [ctx.encrypt(secret_key, [False]) for _ in
              range(output_bits)]
    # [[vm.empty_ciphertext((1,))] for _ in range(output_bits)]
    # andRes = [[vm.empty_ciphertext((1,))] for _ in range(input_bits)]

    for i in range(input_bits):
        andResLeft = [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
        for j in range(input_bits):
            andResLeft[j + i] = vm.gate_and(ctA[j], ctB[i])
            # andResLeft[j + i] = nufhe.LweSampleArray.copy(andRes[j])
        result = addNumbers(andResLeft, result, output_bits)
        result_bits = [ctx.decrypt(secret_key, result[i]) for i in range(size * 2)]
        #print(" nuFHE multiplication intermdiate number is : ",i,boolListToInt(result_bits))

    return result




def predict(ctA,secret_key, output_bits):
    zero = [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
    ctRes = [ctx.encrypt(secret_key, [False]) for _ in range(output_bits)]
    # Copy(ctRes[i], bitResult[0]);
    index=output_bits-1
    ctRes[index] = vm.gate_mux(ctA[index], ctA[index],zero[0])
    one=ctx.encrypt(secret_key, [True])
    ctRes[0]=one
    # Copy(carry[0], bitResult[1])
    return ctRes

if __name__ == '__main__':
   with open('/content/drive/MyDrive/train.txt') as f:
    lines = []
    for line in f:
        lines.append([int(v) for v in line.split()])
    ctx = nufhe.Context()
    secret_key, cloud_key = ctx.make_key_pair()
    vm = ctx.make_virtual_machine(cloud_key)
    size=8
    bits = [[False] for i in range(size - 2)]
    zeros = [[False] for i in range(size)]
    test_size = int(input("Please enter a string:\n"))
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
    w_p=[1,2,3,4,5]

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
    w=[]
    w_b=[]
    for i in range(featuresize-1):
        w.append([[vm.empty_ciphertext((1,))] for i in range(size)])
        w_b.append(fixSizeBoolList(w_p[i],size))
        for j in range(size):
            w[i][j] = ctx.encrypt(secret_key, w_b[i][j])
    bias=[[vm.empty_ciphertext((1,))] for i in range(size*2)]
    bias_b=fixSizeBoolList(w_p[4],size*2)
    for i in range(size*2):
      bias[i]=ctx.encrypt(secret_key, bias_b[i])

    plain_predict=[]
    start_time = time.time()
    for i in range(test_size):
        temp=int(lines[i][0])
        #print(type(temp))
        b_x0=fixSizeBoolList(temp,size)
        temp=int(lines[i][1])
        b_x1=fixSizeBoolList(temp,size)
        temp=int(lines[i][2])
        b_x2=fixSizeBoolList(temp,size)
        temp=int(lines[i][3])
        b_x3=fixSizeBoolList(temp,size)
        ciphertext1=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext2=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext3=[[vm.empty_ciphertext((1,))] for i in range(size)]
        ciphertext4=[[vm.empty_ciphertext((1,))] for i in range(size)]
        for j in range(size):
            ciphertext1[j] = ctx.encrypt(secret_key, b_x0[j])
            ciphertext2[j] = ctx.encrypt(secret_key, b_x1[j])
            ciphertext3[j] = ctx.encrypt(secret_key, b_x2[j])
            ciphertext4[j] = ctx.encrypt(secret_key, b_x3[j])
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
        #predict[i]=[[vm.empty_ciphertext((1,))] for i in range(size*2)]
        #temp1=[]
        #temp2=[]
        presult_mul1 = mulNumbers(ciphertext1,w[0], secret_key, size, size * 2)
        presult_mul2 = mulNumbers(ciphertext2, w[1], secret_key, size, size * 2)
        presult_mul3 = mulNumbers(ciphertext3, w[2], secret_key, size, size * 2)
        presult_mul4 = mulNumbers(ciphertext4, w[3], secret_key, size, size * 2)
        presult_add1 = addNumbers(presult_mul1, presult_mul2,  size*2)
        presult_add2 = addNumbers(presult_mul3, presult_mul4,  size*2)
        presult_add3 = addNumbers(presult_add1, presult_add2,  size*2)
        partial_predict = addNumbers(presult_add3, bias,  size*2)
        predict1=predict(partial_predict,secret_key,size*2)
        predict1.reverse()
        print(predict1)
        print(type(predict1))
        print(type(predict1[0]))
        #result_bits = [ctx.decrypt(secret_key, predict1[i]) for i in range(size*2)]
        result_bits = [ctx.decrypt(secret_key, predict1[i]) for i in range(size*2)]
        plain_predict.append(boolListToInt(result_bits))
        print(" nuFHE multiplication number is : ", boolListToInt(result_bits))
    print("multiplication time",time.time() - start_time)
    #result.reverse()
    for j in range(test_size):
        print(" nuFHE multiplication number is : ", plain_predict[j])

from google.colab import drive
drive.mount('/content/gdrive')
pip freeze --local > /content/gdrive/My\ Drive/colab_installed.txt

import random
import nufhe
import time

def fixSizeBoolList(decimal,size):
    x = [int(x) for x in bin(decimal)[2:]]
    x = list(map(bool, x))
    x = [False]*(size - len(x)) + x
    return x

# in subtraction, ciX have to be greater than ciY
def subtract(ciX, ciY):
    for i in range(size):
        ciXnotTemp = ciX
        a = vm.gate_and(vm.gate_not(ciX), ciY)
        ciX = vm.gate_xor(ciX, ciY)
        aShiftTemp = a
        aShiftTemp.roll(-1, axis=-1)
        ciY = aShiftTemp

    return ciX

def checkSubtract(sub1,sub2):
    if sub1 > sub2:
        return sub2
    else :
        return sub1

def add(ciX, ciY):
# fixed iteration since
    for i in range(size):
        a = vm.gate_and(ciX, ciY)
        b = vm.gate_xor(ciX, ciY)
        aShiftTemp = a
        # using roll as a shift bit
        aShiftTemp.roll(-1, axis=-1)
        ciX = aShiftTemp
        ciY = b

    return b

def boolListToInt(bitlists):
    out = 0
    for bit in bitlists:
        out = (out << 1) | bit
    return out

### testing ###

ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
# size even decimal only
size = 32
# test decimal number. bits lenght need to be less than size/2
deci_x = 3093
deci_y = 1999

x = fixSizeBoolList(deci_x,size)
print(x)
y = fixSizeBoolList(deci_y,size)
print(y)
ciX = ctx.encrypt(secret_key, x)
ciY = ctx.encrypt(secret_key, y)
vm = ctx.make_virtual_machine(cloud_key)

# subtraction have to be done twice since we don't know which one is grater than the other
start_sub = time.time()
subXthenY = ctx.decrypt(secret_key, subtract(ciX, ciY))
subYthenX = ctx.decrypt(secret_key, subtract(ciY, ciX))
# the Lesser subtraction result is the right one, this have to be done after decrypting themessage
plainSubtractNumber = checkSubtract(boolListToInt(subXthenY),boolListToInt(subYthenX))
print("subtraction time",time.time() - start_sub)

#print("time taken by subtact is",elap_sub)
print("reference subtract number is : ", (deci_x-deci_y) ,"/ nuFHE subtract number is : ", plainSubtractNumber)



start_add = time.time()
plainAddNumber = ctx.decrypt(secret_key, add(ciX, ciY))


print("addition time",time.time() - start_add)
print("reference add number is : ", (deci_x+deci_y) ,"/ nuFHE subtract number is : ", boolListToInt(plainAddNumber))





def fixSizeBoolList(decimal,size):
    matrix=[]
    x = [int(x) for x in bin(decimal)[2:]]
    x = list(map(bool, x))
    x = [False]*(size - len(x)) + x
    #matrix = [[x[i]] for i in range(x)]
    pow2 = []
    for i in range(size):
      pow2.append([x[i]])
    return pow2

def listlist(decimal,size):
    matrix = []

    for x in bin(decimal)[2:]:
        matrix.append([])
        #x = [[int(x)] for x in bin(decimal)[2:]]
        matrix[i].append(x)
        x = list(map(bool, x))
        #x = [False]*(size - len(x)) + x
        #matrix[i].append(x)
        #print(matrix)
        #return matrix
        type(x)
    return x

y=fixSizeBoolList(20,size=16)
#z=listlist(20,size=16)
print(y)
print(type(y[0]))
#print(z)

pow2 = []
for x in range(10):
   pow2.append([2 ** x])

print(pow2)

import random
import nufhe
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
size = 32
bits1 = [random.choice([False, True]) for i in range(size)]
bits2 = [random.choice([False, True]) for i in range(size)]
ciphertext1 = ctx.encrypt(secret_key, bits1)
ciphertext2 = ctx.encrypt(secret_key, bits2)
reference = [(b1 ^ b2) for b1, b2 in zip(bits1, bits2)]
vm = ctx.make_virtual_machine(cloud_key)
result = vm.gate_xor(ciphertext1, ciphertext2)
result_bits = ctx.decrypt(secret_key, result)
assert all(result_bits == reference)
print(bits1)
print(bits2)
print(result_bits)



type(bits1)

def ADD(vm, c1, C2,C,cloud_key,thr):
    #C=0
    vm.gate_constant(thr, cloud_key, C, False)
    Sum[len(c1)]
    C_Out[len(c1)]
    for i in c1:
      if i!=0:
        C=C_Out[i-1]
    # Calculating value of sum
      vm.gate_xor(thr, cloud_key, temp1,c1[i],c2[i],perf_params=None )
      vm.gate_xor(thr, cloud_key,Sum[i],temp1,C,perf_params=None )
    # Calculating value of C-Out
      vm.gate_and(thr, cloud_key,temp2, c1[i],c2[i],perf_params=None )
      vm.gate_and(thr, cloud_key,temp3,C,temp1,perf_params=None )
      vm.gate_or(thr, cloud_key,C_Out[i],temp2,temp3,perf_params=None )
    return sum

from reikna.cluda import cuda_api
thr = cuda_api().Thread.create()
ctx = nufhe.Context()
secret_key, cloud_key = ctx.make_key_pair()
size = 32
bits1 = [0,0,0,1,1]
bits2 = [1,0,1,1,1]
C=0
ciphertext1 = ctx.encrypt(secret_key, bits1)
ciphertext2 = ctx.encrypt(secret_key, bits2)
#reference = [not (b1 and b2) for b1, b2 in zip(bits1, bits2)]
vm = ctx.make_virtual_machine(cloud_key)
result=ADD(vm,ciphertext1,ciphertext2,C,cloud_key,thr)
#result = vm.gate_nand(ciphertext1,ciphertext2 )
result_bits = ctx.decrypt(secret_key, result)
#assert all(result_bits == reference)
print(bits1)
print(bits2)
print(result_bits)

def twoscomple(n,nbits):
  a=f"{n & ((1 << nbits) - 1):0{nbits}b}"
  print(a)
twoscomple(1,8)
twoscomple(-1,8)