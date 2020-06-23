import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import time

class Fractal:
    def __init__(self,nproc=6,tol=0.01,maxIter=19,nx=256,ny=256,bbox=[-1,-1,1,1]):
        self.tol = tol
        self.maxIter = maxIter
        self.nx = nx
        self.ny = ny
        self.bbox = bbox
        self.nproc=nproc
                
    def my_newton(self,x0):
        
        dFx = np.array([[ 3*x0[0]**2-3*x0[1]**2 , -6*x0[0]*x0[1] ],[ 6*x0[0]*x0[1] , 3*x0[0]**2-3*x0[1]**2]])
        Fx = np.array([[x0[0]**3-3*x0[0]*x0[1]**2-1.],[3*x0[0]**2*x0[1]-x0[1]**3]])
        dx = np.linalg.solve( dFx , Fx).flatten()

        x = x0-dx

        step = 0
        normA = np.sqrt(dx[0]**2+dx[1]**2)

        while (normA>self.tol):
            dFx = np.array([[ 3*x[0]**2-3*x[1]**2 , -6*x[0]*x[1] ],[ 6*x[0]*x[1] , 3*x[0]**2-3*x[1]**2]])
            Fx = np.array([[x[0]**3-3*x[0]*x[1]**2-1.],[3*x[0]**2*x[1]-x[1]**3]])
            dx = np.linalg.solve( dFx , Fx).flatten()
            x = x - dx

            if (step>self.maxIter):
                break
            normA = np.sqrt(dx[0]**2+dx[1]**2)
            step += 1
        return np.array([x[0],x[1],step])

    def compute(self):
        x0,y0 = np.meshgrid(np.linspace(self.bbox[0],self.bbox[2],self.nx ),np.linspace(self.bbox[1],self.bbox[3],self.ny ))
        x0 = x0.flatten()
        y0 = y0.flatten()
        C = 64*np.ones([self.nx,self.ny],int).flatten()

        xy = np.zeros([self.nx*self.ny,2],float)
        xy[:,0] = x0
        xy[:,1] = y0
        
        pool = multiprocessing.Pool(self.nproc)
        
        print("computing {}x{} pixel on {} proc's ... ".format(self.nx,self.ny,self.nproc), end = '')
        start = time.time()
        sol = np.array(pool.map( self.my_newton,  xy ))
        end = time.time()
        print("runtime = {:5.2f} [s]".format(end - start))

        mask1 = np.where(sol[:,1]<0.1) and np.where(sol[:,1]>-0.1)
        C[mask1] = sol[mask1,2]
        
        mask2 = np.where(sol[:,1]>0.1)
        C[mask2] = sol[mask2,2]+20

        mask3 = np.where(sol[:,1]<-0.1)
        C[mask3] = sol[mask3,2]+40
          
        self.C = C.reshape(self.ny,self.nx)
        
    def get_color_array(self):
        return self.C

def button_press(event):
    coords.append((event.xdata, event.ydata))
   
def button_release(event,img):
    coords.append((event.xdata, event.ydata))
    xmin = np.min((coords[-2][0],coords[-1][0]))
    xmax = np.max((coords[-2][0],coords[-1][0]))
    ymin = np.min((coords[-2][1],coords[-1][1]))
    ymax = np.max((coords[-2][1],coords[-1][1]))

    [x1,y1,x2,y2] = frac.bbox

    nx = frac.nx
    ny = frac.ny
    xmin = x1+xmin*(x2-x1)/nx
    xmax = x1+xmax*(x2-x1)/nx
    ymin = y1+ymin*(y2-y1)/ny
    ymax = y1+ymax*(y2-y1)/ny

    frac.bbox = [xmin,ymin,xmax,ymax]

    frac.compute()
    C = frac.get_color_array()
    
    dpi = 100
    img.set_data(C)
    plt.draw()
    
def main():
    frac = Fractal()

    frac.nx = 400
    frac.ny = 400
    frac.bbox = [-1,-1,1,1]
    frac.nproc = 8
    print("computing {}x{} fractal, using {} processors".format(frac.nx,frac.ny,frac.nproc))
    start = time.time()
    frac.compute()
    end = time.time()
    print("runtime = {:5.2f} [s]".format(end - start))
    
    C = frac.get_color_array()
    dpi = 100
    fig=plt.figure(figsize=(frac.nx/dpi, frac.ny/dpi), dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    img = plt.imshow(C,interpolation='nearest') 
    img.set_cmap('viridis') #'viridis', 'jet', 'hot'
    plt.axis('off')
    
    # imsave("fractal.png", C)
     
    plt.show() 

def main_interactive():
    global frac
    frac = Fractal()
    e = 8
    frac.nx = 2**e
    frac.ny = 2**e
    frac.compute()
    C = frac.get_color_array()
    dpi = 100
    fig=plt.figure(figsize=(frac.nx/dpi, frac.ny/dpi), dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    img = plt.imshow(C,interpolation='nearest')
    plt.axis('off')
    global coords
    coords = []
    
    cid1 = fig.canvas.mpl_connect('button_press_event', button_press)
    cid2 = fig.canvas.mpl_connect('button_release_event', lambda event: button_release(event,img))
     	
    plt.show()


    
if __name__=="__main__":
    main_interactive()
