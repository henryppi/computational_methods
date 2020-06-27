import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy
import sympy

class Crane:
    def __init__(self,
                L_base=1.2,
                W_base=0.5,
                L_boom_a=2.0,
                W_boom_a=0.1,
                L_boom_b=1.0,
                W_boom_b=0.08,
                theta1_0=0.0*np.pi,
                theta2_0=-0.7*np.pi,
                theta_step=5*np.pi/180,
                x_step=0.05,
                Lhyd1=0.7810,
                Lhyd2=0.6510):
                    
        self.L_base = L_base
        self.W_base = W_base
        self.L_boom_a = L_boom_a
        self.W_boom_a = W_boom_a
        self.L_boom_b = L_boom_b
        self.W_boom_b = W_boom_b
        
        self.theta1_0 = theta1_0
        self.theta2_0 = theta2_0
        self.theta_step = theta_step
        self.x_step = x_step

        self.Lhyd1 = Lhyd1
        self.Lhyd2 = Lhyd2
        
        self.theta1 = sympy.symbols('theta1')
        self.theta2 = sympy.symbols('theta2')
                
        self.Tworld = np.eye(4)
        
        self.Tab_sym = sympy.eye(4)
        self.Tab_sym[0:3,0:3] = self.rotz_sym(self.theta1)
        self.Tab_sym[0:3,3] = np.array([0,self.L_base,0])
        
        self.Tbc_sym = sympy.eye(4)
        self.Tbc_sym[0:3,0:3] = self.rotz_sym(self.theta2)
        self.Tbc_sym[0:3,3] = np.array([self.L_boom_a,0,0])
        
        self.Tce = np.eye(4)
        self.Tce[0:3,3] = np.array([self.L_boom_b,0,0])

        self.Tj1 = np.eye(4)
        self.Tj1[0:3,3] = np.array([0.3,0.5*self.L_base,0])

        self.Tj2_sym = sympy.eye(4)
        self.Tj2_sym[0:3,0:3] = self.rotz_sym(self.theta1)
        self.Tj2_sym[0:3,3] = np.array([0.4*L_boom_a,0,0])
        self.Tj2_sym = self.Tab_sym*self.Tj2_sym

        self.Tj3_sym = sympy.eye(4)
        self.Tj3_sym[0:3,0:3] = self.rotz_sym(self.theta1)
        self.Tj3_sym[0:3,3] = np.array([0.6*self.L_boom_a,0,0])
        self.Tj3_sym = self.Tab_sym*self.Tj3_sym

        self.Tj4_sym = np.eye(4)
        self.Tj4_sym[0:3,3] = np.array([0.4*self.L_boom_b,0,0])
        self.Tj4_sym = self.Tab_sym*self.Tbc_sym*self.Tj4_sym

        self.THETA = np.array([[self.theta1_0,self.theta2_0]])
        
        xy = self.compute_forward(self.theta1_0,self.theta2_0)
    
        self.X = np.array([xy])
        
    def append_X(self,xy):
        self.X = np.concatenate([self.X,np.array([xy])],axis=0)

    def append_THETA(self,theta1,theta2):
        self.THETA = np.concatenate([self.THETA, np.array([[theta1,theta2]])],axis=0)
    
    def add_theta_increment(self,q_add):
        self.THETA = np.concatenate([self.THETA, np.array([self.THETA[-1,:]+q_add])],axis=0)
    
    def get_last_X(self):
        return self.X[-1,:]
    
    def get_last_THETA(self):
        return self.THETA[-1,:]
    
    def get_X(self):
        return self.X
    
    def get_THETA(self):
        return self.THETA
    
    def compute_inverse(self,theta1_num,theta2_num,x_dest):
        q = sympy.Matrix([self.theta1,self.theta2])
        Twe = self.Tworld * self.Tab_sym * self.Tbc_sym * self.Tce
        X_sym = sympy.Matrix([[Twe[0,3]],[Twe[1,3]]])
     
        J_sym = X_sym.jacobian(q)

        lam = 0.9 # newton method damping coefficient
        err_tol = 0.001
        m_iter = 0
        max_iter = 100
        
        X_sym_num = X_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        X_sym_num = np.array(X_sym_num).astype(np.float64).flatten()
        
        err = x_dest - X_sym_num
        q_old = np.array([theta1_num,theta2_num])
        q_new = np.array([0,0])
        
        err_mag = np.sqrt(err[0]**2+err[1]**2)

        while (err_mag>=err_tol) & (m_iter<max_iter):
            J = J_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
            J = np.array(J,dtype=np.float64)
            
            delta_theta = np.matrix(np.linalg.pinv(J))*(lam*np.array([err]).T)
            delta_theta = np.array(delta_theta).flatten()
            
            q_new = q_old + np.array(delta_theta,dtype=np.float64)
            q_new = np.array(q_new,dtype=np.float64)

            theta1_num = q_new[0]    
            theta2_num = q_new[1]
            
            q_old = copy.deepcopy(q_new)
            X_sym_num = X_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
            X_sym_num = np.array(X_sym_num).astype(np.float64).flatten()
            
            err = x_dest - X_sym_num
            
            err_mag = np.sqrt(err[0]**2+err[1]**2)
            m_iter +=1
        
        # print('{}/{} newton steps '.format(m_iter,max_iter))
        if m_iter==max_iter:
            print('newton method not converged after {} steps'.format(m_iter))
        return q_new
    
    def compute_forward(self,theta1_num,theta2_num):
        Tab = self.Tab_sym.subs('theta1',theta1_num)
        Tbc = self.Tbc_sym.subs('theta2',theta2_num)
        Twe = self.Tworld * Tab * Tbc * self.Tce
        xy = np.array([Twe[0,3], Twe[1,3]]).astype(np.float64)
        return xy
        
    def rotz(self,theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0.],\
                          [np.sin(theta),  np.cos(theta), 0.],\
                          [  0.,           0.,       1.]])

    def rotz_sym(self,theta):
        return sympy.Matrix([[sympy.cos(theta), -sympy.sin(theta), 0.],\
                          [sympy.sin(theta),  sympy.cos(theta), 0.],\
                          [  0.,           0.,       1.]])   

    def evaluate_key_press(self,w):
        
        bool_q_add = 0
        bool_quit = 0
        
        q_add = np.array([0,0],float)
        x_add = np.array([0,0],float)
        
        if w=='f':
            print('theta1 - \t:',end='')
            q_add[0] = -self.theta_step
            bool_q_add = True
        elif w=='r':
            print('theta1 + \t:',end='')
            q_add[0] = self.theta_step
            bool_q_add = True
        elif w=='g':
            print('theta2 - \t:',end='')
            q_add[1] = -self.theta_step
            bool_q_add = True
        elif w=='t':
            print('theta2 + \t:',end='')
            q_add[1] = self.theta_step
            bool_q_add = True

        elif w=='a':
            print('left \t:',end='')
            x_add[0] = -self.x_step
        elif w=='d':
            print('right \t:',end='')
            x_add[0] = self.x_step
        elif w=='s':
            print('down \t:',end='')
            x_add[1] = -self.x_step
        elif w=='w':
            print('up \t:',end='')
            x_add[1] = self.x_step

        elif w=='q':
            bool_quit = True
        else:
            #do nothing
            pass 

        return q_add,x_add,bool_quit,bool_q_add

    def plot_crane(self,ax,theta1_num,theta2_num):
        
        Tab = self.Tab_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        Tbc = self.Tbc_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        Tj1 = self.Tj1
        Tj2 = self.Tj2_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        Tj3 = self.Tj3_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        Tj4 = self.Tj4_sym.subs({'theta1':theta1_num,'theta2':theta2_num})
        
        X_base = np.array([[-self.W_base, self.W_base, self.W_base,0.1,-0.1,-self.W_base],\
                            [0, 0, 0.1, self.L_base+0.2, self.L_base+0.2, 0.1],\
                            [0, 0, 0,   0,          0,          0]])
                            
        X_boom_A = np.array([[-self.W_boom_a, -self.W_boom_a,0],\
                             [self.W_boom_a+self.L_boom_a, -self.W_boom_a,0],\
                             [self.W_boom_a+self.L_boom_a, self.W_boom_a,0],\
                            [-self.W_boom_a, self.W_boom_a,0]]).T

        X_boom_B = np.array([[-self.W_boom_b, -self.W_boom_b,0],\
                             [self.W_boom_b+self.L_boom_b, -self.W_boom_b,0],\
                             [self.W_boom_b+self.L_boom_b, self.W_boom_b,0],\
                             [-self.W_boom_b, self.W_boom_b,0]]).T
        
        ax = self.plot_patch(ax,X_base,self.Tworld,'r')
        ax = self.plot_patch(ax,X_boom_A,self.Tworld*Tab,'m')
        ax = self.plot_patch(ax,X_boom_B,self.Tworld*Tab*Tbc,'c')

        ax = self.plot_hydraulic_piston_2D(ax,Tj1,Tj2,self.Lhyd1,0.12*self.Lhyd1,'k',2)
        ax = self.plot_hydraulic_piston_2D(ax,Tj3,Tj4,self.Lhyd2,0.15*self.Lhyd2,'k',2)

        ax = self.plot_joint(ax,Tj1,0.05)
        ax = self.plot_joint(ax,Tj2,0.05)
        ax = self.plot_joint(ax,Tj3,0.05)
        ax = self.plot_joint(ax,Tj4,0.05)

        ax = self.plot_joint(ax,self.Tworld*Tab,0.05)
        ax = self.plot_joint(ax,self.Tworld*Tab*Tbc,0.05)
        ax = self.plot_joint(ax,self.Tworld*Tab*Tbc*self.Tce,0.05)
        
        return ax

    def plot_patch(self,ax,X,T,color):
        X1 = np.array(np.matrix(T)*np.concatenate([X,np.ones([1,X.shape[1]])],axis=0))
        poly = Polygon(X1[0:2,:].T, facecolor=color, edgecolor='k')
        ax.add_patch(poly)
        return ax

    def plot_joint(self,ax,T,rad):
        n = 25
        Xa = np.zeros([4,n+1])
        Xa[0,1:] = rad*np.cos(np.linspace(0,0.5*np.pi,n))
        Xa[1,1:] = rad*np.sin(np.linspace(0,0.5*np.pi,n))
        Xa[3,:] = 1
        Xa =np.array(np.matrix(T)*Xa).T
        
        Xb = np.zeros([4,n+1])
        Xb[0,1:] = rad*np.cos(np.linspace(1*np.pi,1.5*np.pi,n))
        Xb[1,1:] = rad*np.sin(np.linspace(1*np.pi,1.5*np.pi,n))
        Xb[3,:] = 1
        Xb =np.array(np.matrix(T)*Xb).T

        Xc = np.zeros([4,n])
        Xc[0,:] = rad*np.cos(np.linspace(0,2*np.pi,n))
        Xc[1,:] = rad*np.sin(np.linspace(0,2*np.pi,n))
        Xc[3,:] = 1
        Xc =np.array(np.matrix(T)*Xc).T
        
        poly1 = Polygon(Xc[:,0:2], facecolor='k', edgecolor='k')
        poly2 = Polygon(Xa[:,0:2], facecolor='w', edgecolor=None)
        poly3 = Polygon(Xb[:,0:2], facecolor='w', edgecolor=None)
        
        ax.add_patch(poly1)
        ax.add_patch(poly2)
        ax.add_patch(poly3)

        return ax

    def plot_hydraulic_piston_2D(self,ax,T1,T2,L0,d,color,lw):
        T1 = np.array(T1)
        T2 = np.array(T2)
        x1 = T1[0:2,3]
        x2 = T2[0:2,3]
        X = np.zeros([2,2],float)
        X[0,:] = T1[0:2,3]
        X[1,:] = np.array(T2[0:2,3])
        
        mag = np.sqrt(np.sum(np.diff(X,axis=0)**2))
        vdir =(X[1,:]-X[0,:])/mag
        vnorm = np.array([vdir[1],-vdir[0]])
        
        x3 = x1 + 0.1*L0*vdir
        x4 = x3+0.5*d*vnorm
        x5 = x3-0.5*d*vnorm
        x6 = x4+0.8*L0*vdir
        x7 = x5+0.8*L0*vdir
        x8 = x2-0.88*L0*vdir
        x9 = x8+0.3*d*vnorm
        x10 = x8-0.3*d*vnorm
        
        px = np.array([[x1[0], x3[0], x3[0], x4[0], x5[0], x2[0], x8[0], x8[0]],\
                       [x3[0], x4[0], x5[0], x6[0], x7[0], x8[0], x9[0], x10[0]]])
        py = np.array([[x1[1], x3[1], x3[1], x4[1], x5[1], x2[1], x8[1], x8[1]],\
                       [x3[1], x4[1], x5[1], x6[1], x7[1], x8[1], x9[1], x10[1]]])
        ax.plot(px,py,'-',color=color,lw=lw)
        return ax

def main():
    global crane
    crane = Crane()
    
    fig = plt.figure(figsize=(5,5)) 
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams['keymap.save'].remove('s')
    plt.rcParams['keymap.grid'].remove('g')
    plt.rcParams['keymap.fullscreen'].remove('f')
    
    theta1_num = 0.0
    theta2_num = -0.7*np.pi
    
    ax = crane.plot_crane(ax,theta1_num,theta2_num)
    plt.xlim([-0.6,3.5])
    plt.ylim([-0.1,4])
    
    def key_press(event,ax):
        wp = event.key
        # sys.stdout.flush()
        q_add,x_add,bool_quit,bool_q_add = crane.evaluate_key_press(wp)

        if bool_q_add:
            crane.add_theta_increment(q_add)
            [theta1_num,theta2_num] = crane.get_last_THETA()
            xy = crane.compute_forward(theta1_num,theta2_num)
            crane.append_X(xy)

        else:
            [theta1_num,theta2_num] = crane.get_last_THETA()
            
            x_dest = crane.get_last_X()+x_add
            q = crane.compute_inverse(theta1_num,theta2_num,x_dest)
            theta1_num = q[0]
            theta2_num = q[1]
            xy = crane.compute_forward(theta1_num,theta2_num)
            crane.append_X(xy)
            crane.append_THETA(theta1_num,theta2_num)

        [theta1_num,theta2_num] = crane.get_last_THETA()
        X = crane.get_X()
        THETA = crane.get_THETA()
        
        print('  theta = [{:1.2f}, {:1.2f}],\t position = [{:1.2f}, {:1.2f} ]'.format(180/np.pi*THETA[-1,0],180/np.pi*THETA[-1,1],X[-1,0],X[-1,1]))
        
        ax.cla()
        ax = crane.plot_crane(ax,theta1_num,theta2_num)
        ax.plot(X[:,0],X[:,1])
        ax.set_xlim([-0.6,3.5])
        ax.set_ylim([-0.1,4])
        fig.canvas.draw()
   
    fig.canvas.mpl_connect('key_press_event', lambda event: key_press(event,ax))
    plt.show()
    
if __name__=="__main__":
    main()
