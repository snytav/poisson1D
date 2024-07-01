import torch

from loss import loss_function

def train(lmb,x_space,y_space,N_epoch,pde,psy_trial,f):
    # lmb = 0.001
    optimizer = torch.optim.SGD(pde.parameters(), lr=lmb)
    import time
    t1 = time.time()
    for i in range(N_epoch):
        #print('begin ',i,loss.item())
        optimizer.zero_grad()
        #print('zero grad ',i,loss.item())
        #print(x_space.device,y_space.device)
        loss = loss_function(x_space, y_space,pde,psy_trial,f)
        #print(loss.device)
        print('{:5d}'.format(i),"   ",'{:15.5e}'.format(loss.item()))
        loss.backward(retain_graph=True)
        #print('loop end ',i,loss.item())
        optimizer.step()
