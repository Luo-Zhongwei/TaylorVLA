A=[1,3,5,6] 
B=[2,4,6,7,10]
ap=0
bp=0
C=[]
while ap<len(A)-1 and bp< len(B)-1:
   if(A[ap]<B[bp]):
       C.append(A[ap])
       ap=ap+1
   elif (A[ap]>=B[bp]):
       C.append(B[bp])
       bp=bp+1
       
len_flag=True

if len(A)>len(B):
    len_flag=True
else:
    len_flag=False
if len_flag:
    C.append(A[ap:])
else:
    C.append(B[bp:])
    
#o(n)
for i in range(len(C)):
    if i==len(C):
        break
    if(C[i]==C[i+1]):
        C.remove[i]