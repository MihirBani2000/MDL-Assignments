ROLLNO = "2019113003"

obs = {
    'rr' : 0.8,
    'gr' : 0.2,
    'gg' : 0.95,
    'rg' : 0.05
}

o = ['r','g','r','g','g','r']
actions = ['r','l','l']
observations = ['g','r','g']
b = [0.33333,0,0.33333,0,0,0.33333]

tableno = int(3)%4 + 1
success_prob = 1-((int(3003)%30+1)/100)
fail_prob = 1-success_prob
print('2019113003 2019101113')
print(success_prob,tableno)
right = [
    [fail_prob,success_prob,0,0,0,0],
    [fail_prob,0,success_prob,0,0,0],
    [0,fail_prob,0,success_prob,0,0],
    [0,0,fail_prob,0,success_prob,0],
    [0,0,0,fail_prob,0,success_prob],
    [0,0,0,0,fail_prob,success_prob],
    ]

left = [
    [success_prob,fail_prob,0,0,0,0],
    [success_prob,0,fail_prob,0,0,0],
    [0,success_prob,0,fail_prob,0,0],
    [0,0,success_prob,0,fail_prob,0],
    [0,0,0,success_prob,0,fail_prob],
    [0,0,0,0,success_prob,fail_prob],
    ]

observed  = 'g'
tsum=0
B = []
for idx in range(3):
    # print('---------------------------------------------')
    observed = observations[idx]
    tsum=0
    if(actions[idx]=='r'):
        for i in range(6):
            qw = observed+o[i]
            # print('UB\'[S'+str(i+1)+'] = '+str(obs[qw])+' * [ ',end='')
            k=0
            for j in range(6):
                k+=right[j][i]*b[j]
                # print("( {:.4f} * {:.4f}) + ".format(right[j][i],b[j]),end='')
            # print(' ] = {:.4f}'.format(k*obs[qw]))
            B.append(k*obs[qw])
            tsum+=k*obs[qw]
        
    else:
        for i in range(6):
            qw = observed+o[i]
            # print('UB\'[S'+str(i+1)+'] = '+str(obs[qw])+'* [ ',end='')
            k=0
            for j in range(6):
                k+=left[j][i]*b[j]
                # print("( {:.4f} * {:.4f}) + ".format(left[j][i],b[j]),end='')
            # print(' ] = {:.4f}'.format(k*obs[qw]))
            B.append(k*obs[qw])
            tsum+=k*obs[qw]

    # print('\nTotal Sum = {:0.4f}'.format(tsum))
    # print("\nOn Normalizing with Total Sum:")
    # print()
    # print("New Beliefs (B\') calculated: \n")
    # for i in range(6):
    #     print("B\'[S{}] = UB\'[S{}]/Total_sum \n \t = {:.4f}/{:.4f} = {:.4f}".format(i+1,i+1,B[i],tsum,B[i]/tsum))
    B = [x/tsum for x in B]
    
    for x in B:
        print(round(x,10), end=' ')
    print()
    b = B
    # print("normalized total sum {}".format(sum(B)))
    B=[]
