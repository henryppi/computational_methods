import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider,Button
import numpy as np
import copy

class Hexapod:
    def __init__(self, rad_base=2,rad_top=1,angle_base=30,angle_top=30):
        self.rad_base = rad_base
        self.rad_top = rad_top
        self.angle_base = angle_base*np.pi/180
        self.angle_top = angle_top*np.pi/180
        
        self.points_base = np.zeros([3,6],float)
        self.points_top = np.zeros([3,6],float)

        alpha= np.array([1./6., 1./6., 3./6., 3./6., 5./6., 5./6.])*(2*np.pi) \
                + self.angle_base*0.5*np.array([-1, 1, -1, 1, -1, 1])
        self.points_base[0,:] = self.rad_base*np.cos(alpha)
        self.points_base[1,:] = self.rad_base*np.sin(alpha)

        alpha= np.array([0, 1./3., 1./3., 2./3., 2./3., 0.])*(2*np.pi) \
                + self.angle_top*0.5*np.array([1, -1, 1, -1, 1, -1])
        self.points_top[0,:] = self.rad_top*np.cos(alpha)
        self.points_top[1,:] = self.rad_top*np.sin(alpha)

    def transformation(self,points_old,x_vec,M):
        points = np.zeros([3,6])
        phi=x_vec[3,0]
        psi=x_vec[4,0]
        chi=x_vec[5,0]
    
        Trotx = np.matrix([[1,0,0],\
                 [0, np.cos(phi),-np.sin(phi)],\
                 [0, np.sin(phi), np.cos(phi)]])
        Troty = np.matrix([[np.cos(psi), 0, np.sin(psi)],\
                        [0, 1, 0],\
                        [-np.sin(psi), 0, np.cos(psi)]])

        Trotz = np.matrix([[np.cos(chi), -np.sin(chi), 0],\
                        [np.sin(chi), np.cos(chi), 0],\
                        [0, 0, 1]])
        ROT=Trotz*Troty*Trotx

        for i in range(6):
            points[:,i] = (ROT*(np.array([points_old[:,i]]).T-M) + M + np.array([x_vec[0:3,0]]).T).flatten()
        return points
        
    def nlgs(self,X,L):
        return  (self.inverse_kinematic(X))**2-L**2

    def partdiff(self,X,L,h=1e-3):
        D = np.zeros([6,6],float)
        for j in range(6):
            xplus=copy.deepcopy(X)
            xplus[j,0]+=h
            xminus=copy.deepcopy(X)
            xminus[j,0]-=h
            fplus = self.nlgs(xplus,L)
            fminus = self.nlgs(xminus,L)
            D[:,j] = (1./(2.*h))*(fplus-fminus)
        return D

    def inverse_kinematic(self,X):
        M = np.zeros([3,1],float) # center of rotation
        points_top = self.transformation(self.points_top,X,M)
        return np.sqrt(np.sum((points_top-self.points_base)**2,axis=0))

    def forward_kinematic(self,X0,L):
        eps = 1e-3
        iStep = 0
        maxStep = 200
        F = np.array([self.nlgs(X0,L)]).T
        D = self.partdiff(X0,L)
        Dinv = np.matrix(np.linalg.inv(D))
        while iStep < maxStep:
            iStep += 1
            X1 = X0 - np.array( Dinv * F)
            X0 = copy.deepcopy(X1)
            F = np.array([self.nlgs(X0,L)]).T
            D = self.partdiff(X0,L)
            Dinv = np.matrix(np.linalg.inv(D))
            res = np.max(F)
            if res<eps:
                iterations = iStep
                iStep = maxStep
                X1 = X0
        
        print('max iterations = ',iterations,' residual = ',res)
        return X1
    
    def update_inverse(self,X):
        X_new = copy.deepcopy(X)
        L_new = self.inverse_kinematic(X_new)
        self.L = L_new
        self.X = X_new
        
    def update_forward(self,X,L):
        L_new = copy.deepcopy(L)
        X_new = self.forward_kinematic(X,L_new)
        self.L = L_new
        self.X = X_new
    
    def set_state(self,X,L):
        self.X = copy.deepcopy(X)
        self.L = copy.deepcopy(L)
    
    def get_state(self):
        return copy.deepcopy(self.X),copy.deepcopy(self.L)
               
    def plot_interactive(self,ax):
        X = self.X
        M = np.zeros([3,1],float)
        B = self.points_base
        T = self.transformation(self.points_top,X,M)
        B = np.concatenate([B,np.array([B[:,0]]).T],axis=1)
        T = np.concatenate([T,np.array([T[:,0]]).T],axis=1)

        ax.cla()
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([0,3])
        ax.plot(B[0,:], B[1,:], B[2,:],'-g',lw=2, label='base')
        ax.plot(T[0,:], T[1,:], T[2,:],'-b',lw=2, label='top')
        for i in range(6):
            iL = np.zeros([3,2])
            iL[:,0] = B[:,i]
            iL[:,1] = T[:,i]
            ax.plot(iL[0,:], iL[1,:], iL[2,:],'o-k', label='legs')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
            
def main():

    class MyState(object):
        state = False
        
        def init(self):
            self.switch([])
            self.switch([])
        
        def switch(self,event):
            self.state = not self.state
            if self.state:
                bswitch.label.set_text('L active, click to activate X')
                # print('forward is set, X deactivated, change L')
                slider_x.eventson = False
                slider_y.eventson = False
                slider_z.eventson = False
                slider_phi.eventson = False
                slider_psi.eventson = False
                slider_chi.eventson = False
                slider_L1.eventson = True
                slider_L2.eventson = True
                slider_L3.eventson = True
                slider_L4.eventson = True
                slider_L5.eventson = True
                slider_L6.eventson = True
            else:
                bswitch.label.set_text('X active, click to activate L')
                # print('inverse is set, L deactivated, change X')
                slider_x.eventson = True
                slider_y.eventson = True
                slider_z.eventson = True
                slider_phi.eventson = True
                slider_psi.eventson = True
                slider_chi.eventson = True
                slider_L1.eventson = False
                slider_L2.eventson = False
                slider_L3.eventson = False
                slider_L4.eventson = False
                slider_L5.eventson = False
                slider_L6.eventson = False
    

    def update_X(val):
        X,L = hex.get_state()
        X[0,0] = slider_x.val
        X[1,0] = slider_y.val
        X[2,0] = slider_z.val
        X[3,0] = slider_phi.val
        X[4,0] = slider_psi.val
        X[5,0] = slider_chi.val
        hex.update_inverse(X)
        X,L = hex.get_state() 

        slider_L1.set_val(L[0])
        slider_L2.set_val(L[1])
        slider_L3.set_val(L[2])
        slider_L4.set_val(L[3])
        slider_L5.set_val(L[4])
        slider_L6.set_val(L[5])
        
        hex.plot_interactive(ax)
    
    def update_L(val):
        X,L = hex.get_state()
        L[0] = slider_L1.val
        L[1] = slider_L2.val
        L[2] = slider_L3.val
        L[3] = slider_L4.val
        L[4] = slider_L5.val
        L[5] = slider_L6.val
        hex.update_forward(X,copy.deepcopy(L))
        X,L = hex.get_state() 
        
        slider_x.set_val(X[0])
        slider_y.set_val(X[1])
        slider_z.set_val(X[2])
        slider_phi.set_val(X[3])
        slider_psi.set_val(X[4])
        slider_chi.set_val(X[5])
        
        hex.plot_interactive(ax)
    
    fig = plt.figure() 
    plt.subplots_adjust(left=0,right=0.5, bottom=0.1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    val = []
    
    hex = Hexapod()
    X = np.array([[0,0,2,0,0,0]],float).T
    L = hex.inverse_kinematic(X)
    hex.set_state(X,L)
   
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.58, 0.90, 0.35, 0.03], facecolor=axcolor) #left bottom width height
    ax_y = plt.axes([0.58, 0.86, 0.35, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.58, 0.82, 0.35, 0.03], facecolor=axcolor)
    ax_phi = plt.axes([0.58, 0.78, 0.35, 0.03], facecolor=axcolor)
    ax_psi = plt.axes([0.58, 0.74, 0.35, 0.03], facecolor=axcolor)
    ax_chi = plt.axes([0.58, 0.70, 0.35, 0.03], facecolor=axcolor)
    
    ax_L1 = plt.axes([0.58, 0.60, 0.35, 0.03], facecolor=axcolor)
    ax_L2 = plt.axes([0.58, 0.56, 0.35, 0.03], facecolor=axcolor)
    ax_L3 = plt.axes([0.58, 0.52, 0.35, 0.03], facecolor=axcolor)
    ax_L4 = plt.axes([0.58, 0.48, 0.35, 0.03], facecolor=axcolor)
    ax_L5 = plt.axes([0.58, 0.44, 0.35, 0.03], facecolor=axcolor)
    ax_L6 = plt.axes([0.58, 0.40, 0.35, 0.03], facecolor=axcolor)
    
    slider_x = Slider(ax_x, 'x', -1., 1., valinit=0.0, valstep=0.025)
    slider_y = Slider(ax_y, 'y', -1., 1., valinit=0.0, valstep=0.025)
    slider_z = Slider(ax_z, 'z', 1.5, 2.5, valinit=2.0, valstep=0.025)
    slider_phi = Slider(ax_phi, 'phi', -np.pi*0.5, np.pi*0.5, valinit=0.0, valstep=0.025*np.pi)
    slider_psi = Slider(ax_psi, 'psi', -np.pi*0.5, np.pi*0.5, valinit=0.0, valstep=0.025*np.pi)
    slider_chi = Slider(ax_chi, 'chi', -np.pi*0.5, np.pi*0.5, valinit=0.0, valstep=0.025*np.pi)
    
    slider_L1 = Slider(ax_L1, 'L1', L[0]*0.66, L[0]*1.5, valinit=L[0]*1.0, valstep=0.05)
    slider_L2 = Slider(ax_L2, 'L2', L[1]*0.66, L[1]*1.5, valinit=L[1]*1.0, valstep=0.05)
    slider_L3 = Slider(ax_L3, 'L3', L[2]*0.66, L[2]*1.5, valinit=L[2]*1.0, valstep=0.05)
    slider_L4 = Slider(ax_L4, 'L4', L[3]*0.66, L[3]*1.5, valinit=L[3]*1.0, valstep=0.05)
    slider_L5 = Slider(ax_L5, 'L5', L[4]*0.66, L[4]*1.5, valinit=L[4]*1.0, valstep=0.05)
    slider_L6 = Slider(ax_L6, 'L6', L[5]*0.66, L[5]*1.5, valinit=L[5]*1.0, valstep=0.05)
    
    ax_button = plt.axes([0.58, 0.20, 0.35, 0.1])
    bswitch = Button(ax_button, 'X active, click to activate L')
    callback = MyState()
    callback.init()
    
    slider_x.on_changed(update_X)
    slider_y.on_changed(update_X)
    slider_z.on_changed(update_X)
    slider_phi.on_changed(update_X)
    slider_psi.on_changed(update_X)
    slider_chi.on_changed(update_X)

    slider_L1.on_changed(update_L)
    slider_L2.on_changed(update_L)
    slider_L3.on_changed(update_L)
    slider_L4.on_changed(update_L)
    slider_L5.on_changed(update_L)
    slider_L6.on_changed(update_L)
    
    bswitch.on_clicked(callback.switch)
   
    slider_z.set_val(X[2])
   
    plt.show()
    
if __name__=="__main__":
    main()
