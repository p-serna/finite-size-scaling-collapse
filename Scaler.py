import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import splrep,splev
from scipy.optimize import minimize

def estimate_bootstrap(x, ycol=2, ecol=3, nbtstrp=100, nqs = 1, quantity=lambda x: x.mean()):
  ''' Function to do bootstraping of a quantity (a function) of the data x, an array with at least two columns ycol (the relevant column that will be bootstrapped) and ecol (error bars). A gaussian kernel with std=error bars is used.
  Input:
    - x: array with shape NxM, and M>=2
    - ycol: column where to add gaussian kernel
    - ecol: column for error bars-> std of g kernel 
    - nbtstrp: # bootstrap steps
    - nqs: # quantities that quantity returns
    - quantity: function of x that returns the quantity that had to be obtained
  Output:
    - res: array nbtstrp x nqs where each row is the result of quantity for each given random sample of x.

  '''
  xt = x*1.0
  dy = 1.0/np.sqrt(xt[:,ecol])
  res = np.zeros((nbtstrp, nqs))
  for i in range(nbtstrp):
    xt[:,ycol] = x[:,ycol]+ np.random.randn(x.shape[0])*dy
    rest = quantity(xt)
    res[i,:] = rest
  return res

BasicScaling_x = lambda d, cf: (d[:,1]-cf[0])*d[:,0]**cf[1]
BasicScalingFSE_x = lambda d, cf: (d[:,1]-cf[0])*d[:,0]**cf[1]/(1+cf[3]/d[:,0]**cf[2])

class Scaler():
  '''
  Raw class to scale data... 

   Input:
    - x: array: 
    - y: Optional
    - dy: Optional
    - L: Optional
    - funx: scaling function for x variable
    - funy: scaling function for y variable
    - fundy:scaling function for errors in y variable
    - cols: Optional - columns of x that match L,x,y,dy
    - errorbars: whether to use errorbars or not
    - emin: minimum tolerance for errorbars
    - knots: # knots for B-splines (it can be changed)

   Example:
    # d: array Nx4 with columns: L,x,y,dy
    
    funx = lambda d, cf: (d[:,1]-cf[0])*d[:,0]**cf[1]
    scaler = Scaler(d,funx = funx,errorbars=True,knots=15)
    minx = scaler.get_opt_cf([0.22522,1.0/0.67])
    print("Optimal values :",minx.x)

    scaler.plot_scaled()
    plt.show()

    dbts = scaler.bootstrap(minx.x,500)
    scaler.plot_bootstrap()


  '''
  def __init__(self, x,y=None,L=None,dy=None,funx=BasicScaling_x,funy=lambda d, cf: d[:,2],fundy=lambda d, cf: d[:,3] ,cols=[0,1,2,3],errorbars=False,emin=1e-16,knots=21):
    self.xf = x
    if y is None:
      try:
        self.y = self.xf[:,cols[2]]
        self.x = self.xf[:,cols[1]]
      except Exception as e:
        print(f"Exception {e}") 
        return
    else:
      self.y = y
      self.x = x
    if L is None:
      self.L = self.xf[:,cols[0]]
    else:
      self.L = L
    if errorbars or dy is not None:
      self.errorbars = True
      if dy is None:
        self.dy = self.xf[:,cols[3]]
      else:
        self.dy = dy
      self.dy = np.clip(self.dy,emin,None)
    else:
      self.errorbars = False

    if errorbars:
      self.d = np.column_stack((self.L,self.x,self.y,
                            1.0/self.dy**2))
    else:
      self.d = np.column_stack((self.L,self.x,self.y,
                            self.x*0+1))
    self.funx = funx 
    self.funy = funy
    self.fundy  = fundy

    self.knots = knots
    self.nknots = None

    self.Ls = list(set(self.L))
    self.Ls.sort()

    self.bts = None
    self.spline = None

  def transform(self,cf):
    self.dsc = self.transform_d(self.d,cf)

    return self.dsc
  
  def transform_d(self,d,cf):
    xt = self.funx(d,cf)
    yt = self.funy(d,cf)
    dyt = self.fundy(d,cf)
    dsc = np.column_stack((self.d[:,0],xt,yt,dyt))
    return dsc

  def clean_knots(self,x,knots):
    '''This function cleans an array of knots, removing those where
    no element of x is present
    '''
    try:
      h = np.histogram(x,bins=knots)[0]
      sel = np.concatenate(([False], h>0))
      sel[-1] = False
    except Exception as e:
      print(f"ERROR {e}")
      print(knots)
      raise ValueError
    return(knots[sel])
    
  def create_spline(self,dscaled=None,knots=None,errors=0):
    '''
      - knots: either integer which generates an array with equidistant knots in the interior, or an array whose points k_i are all xmin<k_i<xmax
    '''
    if dscaled is None:
      xt = 1.0*self.dsc
    else:
      xt = dscaled
      
    xtx, xty = xt[:,1],xt[:,2]
    if errors>0:
      xwy = errors*self.d[:,4]
    else:
      xwy = None
    # To avoid same points at same x values
    xtx += np.random.randn(xtx.shape[0])*(1e-16*np.abs(xtx.max()-xtx.min()))
    sel = xtx.argsort()
    xtx = xtx[sel]
    xty = xty[sel]

    if knots is None:
      knots = self.knots 

    if type(knots) is int:
      # knots have to be inside the range!
      rang = xtx.max()-xtx.min()
      knots = np.linspace(xtx.min()+rang*1e-6,xtx.max()-1e-6*rang,knots)

    knots = self.clean_knots(xtx,knots)
    self.nknots = len(knots)

    try:
      #yspl, f2, ie, ms
      response = splrep(xtx,xty,t=knots,
                               k=3,full_output=1, w= xwy)
    except Exception as e:
      response = 'Exception building splines: {}'.format(e)
      print(response)
    return response

  def chi2(self,cf,d=None,full=False,
            condition = lambda d: np.isfinite(d[:,0]),**kwargs):
    if d is None:
      d = self.d
    dscaled = self.transform_d(d,cf)
    dscaled = dscaled[condition(dscaled),:]
    response = self.create_spline(dscaled,**kwargs)
    self.spline = response[0]
    #yspl, f2, ie, ms = response
    if full:
      return response
    else:
      return response[1]

  def get_opt_cf(self,cf0,condition = lambda d: np.isfinite(d[:,0]),full=False,**kwargs):
    # method="Powell", or others
    obtain_min = lambda d: minimize(lambda cf: self.chi2(cf, d, condition = condition),cf0,**kwargs)
    minx = obtain_min(self.d)
    self.cf = minx.x
    if full:
      return minx, obtain_min

    return minx

  def bootstrap(self,cf0,nbts=100,condition = lambda d: np.isfinite(d[:,0]),**kwargs):
    minx, obtain_min = self.get_opt_cf(cf0,
              condition,full=True, **kwargs)
    obtain_x_chi = lambda minx: [*minx.x,minx.fun*1]
    
    #nqs: # quantities to store in output array: res
    nqs = len(obtain_x_chi(minx))
    res = estimate_bootstrap(self.d,2,3,
                             nbts,nqs=nqs,
                quantity = lambda x: obtain_x_chi(obtain_min(x)))

    self.bts = res
    return res

  def plot_raw(self, ax = None, fmt = "o", **kwargs):
    pass
    return ax

  def plot_scaled(self, cf = None, ax = None, fmt = "o",spfmt = 'k--',condition = lambda d: np.isfinite(d[:,0]), **kwargs):
    if ax is None:
      try: 
        ax = plt.gca()
      except:
        fig,ax = plt.subplots(1)

    if cf is None:
      try:
        dsc = self.dsc
      except:
        dsc = self.transform(self.cf)
    else:
      dsc = self.transform_d(self.d,cf)

    dsc = dsc[condition(dsc),:]
    for L in self.Ls:
      sel = dsc[:,0]==L
      if sel.sum()>0:
        xt = dsc[sel,:]
        #ax.errorbars(xt[:,0],xt[:,1],xt[:,2],fmt,**kwargs)
        ax.plot(xt[:,1],xt[:,2],fmt,label=L,**kwargs)
    
    if self.spline is None:
      response = self.create_spline(dsc)
      self.spline = response[0]
    
    xs = np.linspace(dsc[:,1].min(),dsc[:,1].max(),1001)
    ax.plot(xs,splev(xs,self.spline),spfmt)
    
    
    pass
    return ax

  def plot_bootstrap(self,nfigcols = 3,
                    figax = None, figsize = None):
    if self.bts is None:
      try:
        dbts = self.bootstrap(self.cf)
      except:
        print("Try running at least once optimization")
    else:
      dbts = self.bts    

    ncols = dbts.shape[1]
    nrows = ncols//nfigcols

    if figax is None:
      if figsize is None:
        figsize = (5.5*nfigcols,4.5*nrows)
      fig,axs = plt.subplots(nrows,nfigcols,figsize=figsize)
      axs = axs.ravel()
    else:
      fig,axs = figax
      
    axs[0].plot(dbts[:,0],dbts[:,1],'o',alpha=0.5)
    axs[1].plot(dbts[:,0],dbts[:,2],'o',alpha=0.5)
    axs[2].plot(dbts[:,1],dbts[:,2],'o',alpha=0.5)
    axs[0].set_xlabel("c0",labelpad=20)
    axs[0].set_ylabel("c1")
    axs[1].set_xlabel("c1",labelpad=20)
    axs[2].set_xlabel("c2",labelpad=20)
    axs[1].set_ylabel("chi2")
    axs[2].set_ylabel("chi2")
    plt.tight_layout()

    return fig,axs

if __name__=="__main__":
  xs = np.linspace(0,10,101)
  scaler = Scaler(xs,xs**2,xs*0+1)