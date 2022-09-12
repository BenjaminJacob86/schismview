from glob import glob
import numpy as np
import xarray as xr
import scipy, scipy.interpolate
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import datetime as dt
import matplotlib as mpl
import matplotlib.backends.backend_tkagg as tkagg
import os
import sys
import glob
sys.path.insert(0,'/gpfs/work/ksddata/code/schism/scripts/schism-hzg-utilities/')
from schism import *

#######################
if sys.version_info> (3,0):
    from tkinter import filedialog
    import tkinter as tk
else:
    import Tkinter as tk
    import tkFileDialog as filedialog

class Window(tk.Frame):

    plt.ion()    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)               
        self.master = master
        self.init_window()

    def find_parent_tri(self,tris,xun,yun,xq,yq,dThresh=1000):
        """ parents,ndeweights=find_parent_tri(tris,xun,yun,xq,yq,dThresh=1000)
            find parent for coordinates xq,yq within triangulation tris,xun,yun.
            return: parent triangle ids and barycentric weights of triangle coordinates
        """    
        #% Distance threshold for Point distance
        dThresh=dThresh**2
        
        trisX,trisY=xun[tris],yun[tris]
        trinr=np.arange(tris.shape[0])
        
        #% orthogonal of side vecotrs
        SideX=np.diff(trisY[:,[0, 1, 2, 0]],axis=1)
        SideY=-np.diff(trisX[:,[0, 1, 2, 0]],axis=1)
        
        p=np.stack((xq,yq),axis=1)
        parent=-1*np.ones(len(p),int)
        for ip in range(len(p)):

                dx1=(p[ip,0]-trisX[:,0])
                dy1=(p[ip,1]-trisY[:,0])
                subind=(dx1*dx1+dy1*dy1) < dThresh # preselection
                subtris=trinr[subind]
                
                #% dot products
                parenti=(subtris[ (dx1[subind]*SideX[subind,0] + dy1[subind]*SideY[subind,0] <= 0) \
                               & ((p[ip,0]-trisX[subind,1])*SideX[subind,1] + (p[ip,1]-trisY[subind,1])*SideY[subind,1] <= 0) \
                                 & ( (p[ip,0]-trisX[subind,2])*SideX[subind,2] + (p[ip,1]-trisY[subind,2])*SideY[subind,2] <= 0) ][:])
                if len(parenti):
                    parent[ip]=parenti
        
        # tri nodes
        xabc=xun[tris[parent]]
        yabc=yun[tris[parent]]
        
        # barycentric weights
        divisor=(yabc[:,1]-yabc[:,2])*(xabc[:,0]-xabc[:,2])+(xabc[:,2]-xabc[:,1])*(yabc[:,0]-yabc[:,2])
        w1=((yabc[:,1]-yabc[:,2])*(xq-xabc[:,2])+(xabc[:,2]-xabc[:,1])*(yq-yabc[:,2]))/divisor
        w2=((yabc[:,2]-yabc[:,0])*(xq-xabc[:,2])+(xabc[:,0]-xabc[:,2])*(yq-yabc[:,2]))/divisor
        w3=1-w1-w2
    
        return parent,np.stack((w1,w2,w3)).transpose() 

    def get_node_values(self):	
        self.nodevalues=np.zeros(self.stp.nnodes)	
        self.dryelems=np.zeros(self.stp.nelements)	
        self.u=np.zeros(self.stp.nnodes)	
        self.v=np.zeros(self.stp.nnodes)	
        if  (self.dsi[self.varname].i23d == 2) & (self.dsi[self.varname].ivs == 1): 		
            for nr,file in enumerate(self.files):
            	dsi=xr.open_dataset(file)
            	tmp=dsi[self.varname][self.ti,:,self.lvl].values
            	self.nodevalues[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1]
            	dsi.close()	
       		
        elif  (self.dsi[self.varname].i23d == 1) & (self.dsi[self.varname].ivs == 1): 		
            for nr,file in enumerate(self.files):
            	dsi=xr.open_dataset(file)
            	tmp=dsi[self.varname][self.ti,:].values
            	self.nodevalues[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1]
            	dsi.close()	
        elif  (self.dsi[self.varname].i23d == 1) & (self.dsi[self.varname].ivs == 2): 		
            for nr,file in enumerate(self.files):
            	dsi=xr.open_dataset(file)
            	tmp=dsi[self.varname][self.ti,:,:].values
            	self.u[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1,0]
            	self.v[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1,1]
            	dsi.close()	
            self.nodevalues=np.sqrt(self.u**2+self.v**2)
        elif  (self.dsi[self.varname].i23d == 2) & (self.dsi[self.varname].ivs == 2): 		
            for nr,file in enumerate(self.files):
            	dsi=xr.open_dataset(file)
            	tmp=dsi[self.varname][self.ti,:,self.lvl,:].values
            	self.u[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1,0]
            	self.v[self.node_map[nr][:,1]-1]=tmp[self.node_map[nr][:,0]-1,1]
            	dsi.close()	
            self.nodevalues=np.sqrt(self.u**2+self.v**2)				
        #for nr,file in enumerate(self.files):
        #	dsi=xr.open_dataset(file)
        #	tmp=dsi['wetdry_elem'][self.ti,:].values
        #	self.dryelems[self.elem_map[nr][:,1]-1]=tmp[self.elem_map[nr][:,0]-1]
        #	dsi.close()	
			
    def schism_plotAtelems(self,nodevalues):
        ph=plt.tripcolor(self.stp.x,self.stp.y,self.faces[:,:3],facecolors=self.nodevalues[self.faces[:,:3]].mean(axis=1),shading='flat',cmap=plt.cm.jet)# shading needs gouraud to allow correct update
        ch=plt.colorbar()
        plt.tight_layout()
        return ph,ch

    def schism_updateAtelems(self):
        plt.figure(1)
        do_quiv=self.quivVar.get() and (self.dsi[self.varname].ivs == 2)
        print('quiver:'+str(do_quiv))			
        if self.quiver!=0:
                self.quiver.remove()
                self.arrowlabel.remove()    
                
        self.get_node_values()
        title=self.varname
        #u=self.nodevalues=weights[0,:]*self.ncv[self.varname][self.ti,:,:,0][self.nodeinds,ibelow]+weights[1,:]*self.ncv[self.varname][self.ti,:,:,0][self.nodeinds,iabove]
        #        v=self.nodevalues=weights[0,:]*self.ncv[self.varname][self.ti,:,:,1][self.nodeinds,ibelow]+weights[1,:]*self.ncv[self.varname][self.ti,:,:,1][self.nodeinds,iabove]
         #   self.nodevalues=np.sqrt(u*u+v*v)
         #   do_quiv=True and self.quivVar.get()
         #   title='abs ' + self.varname    
            

        if self.CheckEval.get()!=0: # evaluate on displayed variable
                expr= self.evalex.get()
                expr=expr[expr.index('=')+1:].replace('x','self.nodevalues').replace('A','self.A').replace('dt','self.dt')
                self.nodevalues=eval(expr)    
                
        if self.varname != 'depth':
            #title+=' @ ' + str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))) + ' level= ' + str(lvl)
            title+=' @ ' + ' level= ' + str(self.lvl)
        # setting colorbar
        #self.elemvalues=np.ma.masked_array(self.nodevalues[self.faces[:,:3]].mean(axis=1),mask=self.dryelems[self.stp.nvplt]==1)
        self.elemvalues=np.ma.masked_array(self.nodevalues[self.faces[:,:3]].mean(axis=1),mask=None)
        
        self.ph.set_array(self.elemvalues) 
        if not self.CheckVar.get():
            self.clim=(np.nanmin(self.elemvalues),np.nanmax(self.elemvalues))
            self.minfield.delete(0,tk.END)
            self.maxfield.delete(0,tk.END)
            self.minfield.insert(8,str(np.nanmin(self.elemvalues)))
            self.maxfield.insert(8,str(np.nanmax(self.elemvalues)))
        else:
            self.clim=(np.double(self.minfield.get()),np.double(self.maxfield.get()))
        self.ph.set_clim(self.clim) 
        
        # add quiver
        if do_quiv:
            n=100 # arrows shown along one axis
            xlim=plt.xlim()
            ylim=plt.ylim()
            x=np.arange(xlim[0],xlim[1],(xlim[1]-xlim[0])/n)
            y=np.arange(ylim[0],ylim[1],(ylim[1]-ylim[0])/n)
            X, Y = np.meshgrid(x,y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            d,qloc=self.xy_nn_tree.query(positions.transpose())
            xref,yref=np.asarray(plt.axis())[[1,2]] +  np.diff(plt.axis())[[0,2]]*[- 0.2, 0.1]
            vmax=np.double(self.maxfield.get())
            self.quiver=plt.quiver(np.concatenate((self.x[qloc],(xref,))),np.concatenate((self.y[qloc],(yref,))),np.concatenate((self.u[qloc],(vmax,))),np.concatenate((self.v[qloc],(0,))),scale=2*vmax,scale_units='inches') #self.maxfield.get()
            self.arrowlabel=plt.text(xref,yref,'\n'*3+str(self.maxfield.get())+' m/s')
            print('done quiver')
        else:
            self.quiver=0
            
        # remove annotations if figure 2 closed
        if 2 not in plt.get_fignums() and (len(self.anno)>0) :
            for item in self.anno:
                item.remove()
            self.anno=[]
        
        #update plot    
        plt.title(title)
        self.update_plots()
        print("done plotting ")

    def update_plots(self):
        for figi in plt.get_fignums():
            plt.figure(figi).canvas.draw()
            plt.figure(figi).canvas.flush_events() # flush gui events
    
    def init_window(self):
        self.pack(fill=tk.BOTH,expand=1)

		# load setup  
        print("naviagate into schsim run directory (containing hgrid, vgrid)")
        self.runDir=filedialog.askdirectory(title='enter run directory direcory')+'/'
        print('loading setup') 
        os.chdir(self.runDir)
        self.stp=schism_setup()
        self.x=np.asarray(self.stp.lon)
        self.y=np.asarray(self.stp.lat)
        print('done loading setup')
		
        # load files    
        print("naviagate into outputs directory")
        self.combinedDir=filedialog.askdirectory(title='enter schout_*.nc direcory')+'/'
        
        # load mapping files
        mapping_files=np.sort(glob.glob(self.combinedDir+'local_to_global_????'))
        self.elem_map=[]
        self.node_map=[]
        for nr,file in enumerate(mapping_files):
        	self.elem_map+=[[]]
        	self.node_map+=[[]]
        	f=open(file)
        	for i in range(3):
        		line=f.readline()
        	nrelems=np.int(line)
        	for i in range(nrelems):	
        		self.elem_map[nr]+=[[np.int(item) for item in f.readline().split() ]] 
        	self.elem_map[nr]=np.asarray(self.elem_map[nr])
        	line=f.readline()
        	nrnodes=np.int(line)
        	for i in range(nrnodes):	
        		self.node_map[nr]+=[[np.int(item) for item in f.readline().split() ]] 
        	self.node_map[nr]=np.asarray(self.node_map[nr])	
        	f.close()

        	# grid node -> file -> node in file
        	self.node_proc_map=np.zeros((0,3),int)
        	for nr in range(len(self.node_map)):
        		tmp=np.zeros((len(self.node_map[nr]),3),int)
        		tmp[:,0]=self.node_map[nr][:,1]
        		tmp[:,1]=nr
        		tmp[:,2]=self.node_map[nr][:,0]
        		self.node_proc_map=np.vstack((self.node_proc_map,tmp))			

		# stack files
        self.procfiles=glob.glob(self.combinedDir+'schout_0000_'+'*.nc')
        nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.procfiles]
        nrs=list(np.asarray(nrs)[np.argsort(nrs)])		
        self.files=glob.glob(self.combinedDir+'schout_????_'+str(nrs[0])+'.nc')
        procnrs=[int(file[file.find('schout')+7:file.rfind('_')]) for file in self.files]
        self.files=list(np.asarray(self.files)[np.argsort(procnrs)])
        procnrs=list(np.asarray(procnrs)[np.argsort(procnrs)])		
        self.nstacks=len(self.procfiles)
        #print('found ' +str(len(nrs)) +' files')

        # file access        
        self.dsi=xr.open_dataset(self.procfiles[0])
        self.nt=len(self.dsi['time'])
        self.nnodes=len(self.stp.x)
        self.nz=len(self.stp.vgrid[1])

        #self.reftime=dt.datetime.strptime(self.nc['time'].units[14:33],'%Y-%m-%d %H:%M:%S')
                                            
        
        #try:
        #    lmin = self.ncv['node_bottom_index'][0:]
        #except:
        #    zvar = self.ncv['zcor'][0]
        #    lmin = np.zeros(self.x.shape,dtype='int')
        #    for i in range(len(self.x)):
        #        try:
        #            lmin[i] = max(np.where(zvar.mask[i])[0])
        #        except:
        #            lmin[i] = 0
        #            lmin = lmin+1
        
        # next neighbour node look up tree
        self.xy_nn_tree = cKDTree([[self.stp.lon[i],self.stp.lat[i]] for i in range(len(self.stp.x))])
        #self.minavgdist=np.min(np.sqrt(np.abs(np.diff(self.x[self.faces[:,[0,1,2,0]]],axis=1))**2+np.abs(np.diff(self.y[self.faces[:,[0,1,2,0]]],axis=1))**2).mean(axis=1))

        # mesh for mesh visualisazion        
        xy=np.c_[self.stp.x,self.stp.y]
        self.faces=self.stp.nvplt#np.asarray(self.stp.nv)-1
        #self.mesh_tris=xy[self.faces[np.where(self.faces.mask.sum(axis=1)),:3]][0,:]
        #self.mesh_quads=xy[self.faces[np.where(self.faces.mask.sum(axis=1)==0),:4]][0,:]
        #self.tripc = PolyCollection(self.mesh_tris,facecolors='none',edgecolors='k',linewidth=0.2) #, **kwargs)
        #self.hasquads = np.min(self.mesh_quads.shape)>0
        
        #if self.hasquads: #build tri only grid for faster plotting
        #    self.quadpc = PolyCollection(self.mesh_quads,facecolors='none',edgecolors='r',linewidth=0.2) #, **kwargs)    
        #    self.faces  
        #    print("building pure triangle grid for easier plotting")
        #    faces2=[]
        #    self.origins=[] # mapping from nodes to triangles for triplot
        #    for nr,elem in enumerate(self.faces):
        #        if elem.mask.sum()==1:
        #            faces2.append(elem[:3])
        #            self.origins.append(nr)
        #        else: # split quad into tris
        #            faces2.append(elem[[0,1,2]])
        #           faces2.append(elem[[0,2,3]])
        #            self.origins.append(nr)
        #            self.origins.append(nr)
        #    self.faces=np.array(faces2)                    
        #    self.origins=np.array(self.origins)
        #    print("done")
        #else:
        #     self.faces=self.faces[:,:3]
        #     self.origins=np.arange(self.faces.shape[0])
        ##########################################  

       # load variable list from netcdf #########################################            
        exclude=['time','SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes','SCHISM_hgrid_node_x',     'SCHISM_hgrid_node_y', 'node_bottom_index','SCHISM_hgrid_face_x', 'SCHISM_hgrid_face_y', 'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y', 'edge_bottom_index','sigma', 'dry_value_flag','coordinate_system_flag', 'minimum_depth', 'sigma_h_c', 'sigma_theta_b','sigma_theta_f','sigma_maxdepth', 'Cs', 'wetdry_elem'] # exclude for plot selection
        varlist=[]
        for vari in self.dsi.keys():
            if vari not in exclude:
                varlist.append(vari)
        ########################################################################## 
        
        
        #### GUI ELEMENTS ##############
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.client_exit)
        menubar.add_cascade(label="File", menu=filemenu)
        saveMenu = tk.Menu(self.master)

        #filemenu.add_cascade(label="Save", menu=saveMenu)
        #saveMenu.add_command(label="CurrentNodeData", command=self.savenodevalues)
        #saveMenu.add_command(label="Coordinates", command=self.savecoords)
        #saveMenu.add_command(label="Timeseries", command=self.savets)
        #saveMenu.add_command(label="Hovmoeller", command=self.savehov)
        #saveMenu.add_command(label="Profiles", command=self.profiles)
        #saveMenu.add_command(label="Transect", command=self.transect_callback)

        extractmenu = tk.Menu(menubar, tearoff=0)
        extractmenu.add_command(label="Timeseries", command=self.timeseries)
        extractmenu.add_command(label="Hovmoeller", command=self.hovmoeller)
        extractmenu.add_command(label="Profiles", command=self.profiles)
        extractmenu.add_command(label="Transect", command=self.transect_callback)
        menubar.add_cascade(label="Extract", menu=extractmenu)
        self.master.config(menu=menubar)

        row=0
        L1=tk.Label(self,text='variable')
        L1.grid(row=row,column=0)#.place(x=260,y=0)

        L2=tk.Label(self,text='layer')
        L2.grid(row=row,column=1)#.place(x=200,y=0)

        L3=tk.Label(self,text='stack')
        L3.grid(row=row,column=2)

        L2=tk.Label(self,text='timestep')
        L2.grid(row=row,column=3)#.place(x=140,y=0)

        row+=1
        self.variable = tk.StringVar(self)
        self.variable.set('depth') # default value
        self.variable.trace("w",self.variable_callback)
        w3=tk.OptionMenu(self, self.variable, *['depth']+varlist) #w3.config(width=10)
        w3.grid(row=row,column=0)

        levels=range(self.dsi['zcor'].shape[-1])[::-1]
        self.lvl_tk = tk.IntVar(self)
        self.lvl_tk.set(levels[0]) # default value
        self.lvl_tk.trace("w",self.lvl_callback)
        self.lvl=levels[0]
        w3=tk.OptionMenu(self, self.lvl_tk, *levels)
        w3.grid(row=row,column=1)

        self.stacks=nrs
        self.stack = tk.IntVar(self)
        self.stack.set(nrs[0]) 
        self.stack.trace("w",self.stack_callback)
        self.current_stack=nrs[0]
        w=tk.Spinbox(self, values=nrs, width=5,validate='all',textvariable=self.stack) # try command and set values in function with checking
        w.grid(row=row,column=2)

        self.ti_tk = tk.IntVar(self)
        self.ti_tk.set(0) # default value
        self.ti_tk.trace("w",self.ti_callback)
        self.ti=0
        w2=tk.OptionMenu(self, self.ti_tk, *range(len(self.dsi['time'])))
        w2.grid(row=row,column=3)

        row+=1
        start = tk.Button(self,text='|<<',command=self.firstts)
        start.grid(row=row,column=0)
        play = tk.Button(self,text='play',command=self.playcallback)
        play.grid(row=row,column=1)
        self.play=True
        stop = tk.Button(self,text='stop',command=self.stopcallback)
        stop.grid(row=row,column=2)
        end = tk.Button(self,text='>>|',command=self.lastts)
        end.grid(row=row,column=3)

        row+=1 # z  lvel interpolation
        self.CheckFixZ = tk.IntVar(value=0)
        fixz=tk.Checkbutton(self,text='interp to z(m):',variable=self.CheckFixZ)
        fixz.grid(sticky = tk.W,row=row,column=0)
        self.fixdepth=tk.Entry(self,width=8)
        self.fixdepth.grid(row=row,column=1)

        row+=1 # eval results
        self.CheckEval = tk.IntVar(value=0)
        fixz=tk.Checkbutton(self,text='eval  :',variable=self.CheckEval)
        fixz.grid(sticky = tk.W,row=row,column=0)
        self.evalex=tk.Entry(self,width=16)
        self.evalex.grid(row=row,column=1,columnspan=2)
        self.evalex.insert(8,'x=x')

        row+=1 # Apperance
        h1=tk.Label(self,text='\n Appearance:',anchor='w',font='Helvetica 10 bold')
        h1.grid(row=row,column=0)

        row+=1 # Clorors 
        self.CheckVar = tk.IntVar(value=0)
        fixbox=tk.Checkbutton(self,text='fix colorbar',variable=self.CheckVar)
        fixbox.grid(sticky = tk.W,row=row,column=0)

        # colormap
        maps=np.sort([m for m in plt.cm.datad if not m.endswith("_r")])
        self.cmap = tk.StringVar(self)
        self.cmap.trace("w",self.cmap_callback)
        w4=tk.OptionMenu(self, self.cmap, *maps)
        w4.grid(row=row,column=1)
        self.cmap.set('jet') # default value

        row+=1
        l4=tk.Label(self,text='caxis:')
        l4.grid(row=row+1,column=0)
        l5=tk.Label(self,text='min:')
        l5.grid(row=row,column=1)
        self.minfield=tk.Entry(self,width=8)
        self.minfield.grid(row=row+1,column=1)
        l6=tk.Label(self,text='max:')
        l6.grid(row=row,column=2)
        self.maxfield=tk.Entry(self,width=8)
        self.maxfield.grid(row=row+1,column=2)

        row+=2
        self.xminvar = tk.StringVar()
        self.xminvar.trace("w",self.updateaxlim)
        self.xmaxvar = tk.StringVar()
        self.xmaxvar.trace("w",self.updateaxlim)
        self.yminvar = tk.StringVar()
        self.yminvar.trace("w",self.updateaxlim)
        self.ymaxvar = tk.StringVar()
        self.ymaxvar.trace("w",self.updateaxlim)
        
        self.zminvar = tk.StringVar()
        self.zminvar.trace("w",self.updatezlim)
        self.zmaxvar = tk.StringVar()
        self.zmaxvar.trace("w",self.updatezlim)
        
        l=tk.Label(self,text='xaxis:')
        l.grid(row=row,column=0)
        self.xminfield=tk.Entry(self,width=8,textvariable=self.xminvar)
        self.xminfield.grid(row=row,column=1)
        self.xmaxfield=tk.Entry(self,width=8,textvariable=self.xmaxvar)
        self.xmaxfield.grid(row=row,column=2)
        
        row+=1
        l=tk.Label(self,text='yaxis:')
        l.grid(row=row,column=0)
        self.yminfield=tk.Entry(self,width=8,textvariable=self.yminvar)
        self.yminfield.grid(row=row,column=1)
        self.ymaxfield=tk.Entry(self,width=8,textvariable=self.ymaxvar)
        self.ymaxfield.grid(row=row,column=2)
        
        row+=1
        l=tk.Label(self,text='zaxis:')
        l.grid(row=row,column=0)
        self.zminfield=tk.Entry(self,width=8,textvariable=self.zminvar)
        self.zminfield.grid(row=row,column=1)
        self.zmaxfield=tk.Entry(self,width=8,textvariable=self.zmaxvar)
        self.zmaxfield.grid(row=row,column=2)

        row+=1
        self.meshVar = tk.IntVar(value=0)
        meshbox=tk.Checkbutton(self,text='show mesh',variable=self.meshVar,command=self.mesh_callback)
        meshbox.grid(sticky = tk.W,row=row,column=0)

        self.quivVar = tk.IntVar(value=0)
        quivbox=tk.Checkbutton(self,text='show arrows',variable=self.quivVar)
        quivbox.grid(sticky = tk.W,row=row,column=1)

        row+=1
        self.stream = tk.IntVar(value=0)
        imstream=tk.Checkbutton(self,text='stream to images :',variable=self.stream)
        imstream.grid(sticky = tk.W,row=row,column=0)
        self.picture_dir_set=False

        row+=1
        h2=tk.Label(self,text='\n Extract:       ',anchor='w',font='Helvetica 10 bold')
        h2.grid(row=row,column=0)
        
        row+=1
        l7=tk.Label(self,text='extract from:')
        l7.grid(row=row,column=0)
        self.exfrom=tk.Entry(self,width=8)
        self.exfrom.grid(row=row+1,column=0)
        self.exfrom.insert(8,'0')
        l8=tk.Label(self,text='extract until:')
        l8.grid(row=row,column=1)
        self.exto=tk.Entry(self,width=8)
        self.exto.grid(row=row+1,column=1)
        self.exto.insert(8,str((self.stacks[self.nstacks-1]-self.stacks[0]+1)*self.nt))
        

        # initial plot
        self.varname='depth'
        self.nodevalues=np.asarray(self.stp.depths) 
        self.shape=self.nodevalues.shape
        self.plot=plt.figure(1)
        print("plotting " + str(self.varname))                      
        self.clim=(np.min(self.nodevalues),np.max(self.nodevalues))       
        self.fix_clim=False
        #self.ph,self.ch=self.schism_plotAtelems(self.nodevalues)
        self.ph,self.ch=self.stp.plotAtelems(np.asarray(self.stp.depths)[self.stp.nvplt].mean(axis=1))
        plt.title(self.varname)
        self.quiver=0
        self.mesh_plot=0
        self.anno=[]
        self.minfield.insert(8,str(self.nodevalues.min()))
        self.maxfield.insert(8,str(self.nodevalues.max()))
        self.update_plots()        # on unix cluster initial fiugre remains black -> therefore
        self.update_plots()        # on unix cluster initial fiugre remains black -> therefore
        print("done initializing")                      

        row+=1    
        # insert axlimits
        self.xminfield.insert(8,str(plt.gca().get_xlim()[0]))
        self.xmaxfield.insert(8,str(plt.gca().get_xlim()[1]))
        self.yminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.ymaxfield.insert(8,str(plt.gca().get_ylim()[1]))
    
        #tkagg.NavigationToolbar2TkAgg(self.plot.canvas, self.master)

    ####### call back
    def variable_callback(self,*args):
        # select variable
        print("selected variable " + self.variable.get())
        self.varname=self.variable.get()
        if self.varname!='depth':
            self.ivs=self.dsi[self.varname].ivs
        else:
            self.ivs=1
        self.schism_updateAtelems()

        #update extract plots in figure 2
        if 2 in plt.get_fignums():
                self.extract(coords=self.coords)   

    def lvl_callback(self,*args):         # set level    
        self.lvl=self.lvl_tk.get()
        print("selected level" +  str(self.lvl))
        self.schism_updateAtelems()

    def stack_callback(self,*args): # load new stack
        stacknow=self.stack.get()
        if stacknow!=self.current_stack:
            if  self.stacks[self.nstacks-1] < stacknow or stacknow< self.stacks[0]:
                print('stack not existent')
            else:
			
                self.files=glob.glob(self.combinedDir+'schout_????_'+str(stacknow)+'.nc')
                procnrs=[int(file[file.find('_')+1:file.rfind('_')]) for file in self.files]
                self.files=list(np.asarray(self.files)[np.argsort(procnrs)])

                #self.file=self.combinedDir+'schout_'+str(stacknow)+'.nc' 
                #self.files=glob.glob(self.combinedDir+'schout_????_'+str(stacknow)+'.nc')
				
                print("loading files of stack" +  str(self.stack))
                self.ti_tk.set(0) 
                self.ti=0
                self.current_stack=stacknow
                #self.times=self.ncv['time'][:]

    def ti_callback(self,*args): #set timestep
        self.ti=self.ti_tk.get()
        print("selected timestep " +  str(self.ti))
        #self.dryelems=self.ncv['wetdry_elem'][self.ti,:][self.origins]
        self.schism_updateAtelems()
        if 2 in plt.get_fignums() and (self.extract==self.profiles or self.extract==self.transect_callback):
                self.extract(self.coords)

    def playcallback(self,*args):
            count=(self.current_stack-self.stacks[0])*self.nt+self.ti_tk.get()
            if count < self.nt*len(self.files)-1 and self.play:
                self.nextstep()
                if self.stream.get():
                    self.stream2image()
            else:
                self.play=True
                return
            self.master.after(60,self.playcallback) # create recursive play loop

    def stopcallback(self,*args):
         self.play=False          
                            
    def get_layer_weights(self,dep): 
        print('calculating weights for vertical interpolation')
        ibelow=np.zeros(self.nnodes,int)
        iabove=np.zeros(self.nnodes,int)
        weights=np.zeros((2,self.nnodes))

        zcor=self.ncv['zcor'][self.ti_tk.get(),:,:]
        a=np.sum(zcor<=dep,1)
        ibelow=a+np.sum(zcor.mask,1)-1
        iabove=np.minimum(ibelow+1,self.nz-1)
        inodes=np.where(a>0)[0]
        ibelow2=ibelow[inodes]
        iabove2=iabove[inodes]

        d2=zcor[inodes,iabove2]-dep
        d1=dep-zcor[inodes,ibelow2]
        ivalid=d1>0.0
        iset=d1==0.0
        d1=d1[ivalid]
        d2=d2[ivalid]

        weights[0,inodes[ivalid]]=1/d1/(1/d1+1/d2)
        weights[1,inodes[ivalid]]=1/d2/(1/d1+1/d2)
        weights[0,inodes[iset]]=1
        weights[:,np.sum(weights,0)==0.0]=np.nan
        
        return ibelow, iabove, weights

                
    def stream2image(self,*args):
        if self.picture_dir_set==False:
            self.streamdir=filedialog.askdirectory(title='select output directory for image stream')+'/'
            self.picture_dir_set=True
        plt.figure(1)
        plt.savefig(self.streamdir+'{0:05d}'.format((self.stack.get()-1)*self.nt+self.ti)+'_'+self.varname+'.png',dpi=300)
            
          
    def cmap_callback(self,*args):   
        cmap=plt.cm.get_cmap(self.cmap.get())
        cmap.set_bad('grey',1)
        for i in reversed(plt.get_fignums()):
               plt.figure(i) 
               plt.set_cmap(cmap)
        self.update_plots() #self.plot.canvas.draw() #self.update_plots()
        
        
    def mesh_callback(self,*args):        
        """ add mesh to plot """
        plt.figure(1)
        if (self.meshVar.get()==1) and (self.mesh_plot==0):
            self.mesh_plot=1
            plt.gca().add_collection(self.tripc)
            if self.hasquads:
                plt.gca().add_collection(self.quadpc)
        elif  (self.meshVar.get()==0) and (self.mesh_plot!=0):
            self.tripc.remove()
            if self.hasquads:
                self.quadpc.remove()
            self.mesh_plot=0
        self.update_plots()  
      
    def ask_coordinates(self,n=-1):
            
            if n==-1:
                plt.title("click coordinates in Fig.1. Press ESC when finished")
                print("click coordinates in Fig.1. Press ESC when finished")
            else:
                plt.title("click coordinate in Fig.1.")
                print("click coordinate in Fig.1.")
            self.update_plots()
            plt.figure(1)
            self.coords=plt.ginput(n,show_clicks='True')
#            plt.title(self.varname+' @ ' + str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))) + ' level= ' + str(self.lvl))
            plt.title(self.varname+' @ ' + ' level= ' + str(self.lvl))
            # plot coordinates on main figure / remove potential old coordinates
            if (len(self.anno)>0) :
                for item in self.anno:
                    item.remove()
            
            # interpolate coordinates for transect
            if self.extract==self.transect_callback: # interp coordinates
                xy=np.asarray(self.coords)
                dxdy=np.diff(xy,axis=0)
                dr=np.sqrt((dxdy**2).sum(axis=1))
                r=dxdy[:,1]/dxdy[:,0]
                coords1=[] 
                for i in range(len(self.coords)-1):
                    dx=np.linspace(0,dxdy[i,0],int(np.floor(dr[i]/self.minavgdist)))
                    coords1+=([(xi[0][0],xi[0][1]) for xi in zip((xy[i,:]+np.stack((dx, dx*r[i]),axis=1)))])
                xy=np.asarray(coords1)
                
                self.minterp='nn' #'bary_sigma'   #'inv_dist10' # richards approach  #self.minterp='bary'
                if self.minterp=='nn':
                    d,self.nn=self.xy_nn_tree.query(coords1)  # nearest node
                    nn,iunique=np.unique(self.nn,return_index=True)
                    iunique=np.sort(iunique)
                    self.nn=self.nn[iunique]
                    self.npt=len(self.nn)
                    self.coords=([ (self.x[nni],self.y[nni]) for nni in nn])
                    x,y=self.x[self.nn],self.y[self.nn]
                    #x,y=xy[iunique,0],xy[iunique,1]
                elif 'bary' in self.minterp:
                    print(xy[:,0])
                    self.parents,self.ndeweights=self.find_parent_tri(self.faces,self.x,self.y,xy[:,0],xy[:,1],dThresh=1000)
                    nn,iunique=np.unique(self.parents,return_index=True)
                    iunique=np.sort(iunique)
                    self.parents=self.parents[iunique]
                    self.ndeweights=self.ndeweights[iunique,:]
                    self.npt=len(self.parents)
                    x,y=xy[iunique,0],xy[iunique,1]
                    self.nn=self.faces[self.parents].flatten()
                    self.wtrans=np.tile(self.ndeweights,(58,1,1)).swapaxes(0,2)
                    self.wtransvec=np.tile(self.wtrans,(2,1,1,1)).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3) #.shape
                    self.coords=coords1
                elif self.minterp=='inv_dist10':
                    self.dist,self.inds=self.xy_nn_tree.query(list(zip(xy[:,0],xy[:,1])),k=10)
                    self.wtrans=1.0/self.dist**2
                    x,y=xy[:,0],xy[:,1]
                    self.npt=len(x)
                    
                self.anno=plt.plot(x,y,'k.-')        
                self.anno.append(plt.text(x[0],y[0],'P '+str(0)))
                self.anno.append(plt.text(x[-1],y[-1],'P '+str(self.npt-1)))
                    
                    
            else:    
                d,self.nn=self.xy_nn_tree.query(self.coords)  # nearest node
                self.npt=len(self.nn)
                self.anno=plt.plot(self.x[self.nn],self.y[self.nn],'k+')             
                for i,coord in enumerate(self.coords):
                    xi,yi=coord
                    self.anno.append(plt.text(xi,yi,'P '+str(i)))
            self.update_plots()
            
            if self.npt==1:
                self.nn=self.nn[0]


      
    def timeseries(self,coords=None):
        """"
        Extract timeseries at nextneighbours to clicked coordinates
        """
        
        self.extract=self.timeseries # identify method
        if coords==None:
            self.ask_coordinates()

        print('extracting timeseries for ' + self.varname + ' at coordinats: ' + str(self.coords))
        i0,i1=int(self.exfrom.get()),int(self.exto.get())

        if self.npt==1:
            self.nn=[self.nn]		            	
        if self.ivs==1:
            self.ts=np.zeros((i1-i0+1,self.npt))	
        else:
            self.ts=np.zeros((i1-i0+1,len(self.nn),2))	
        i0=np.int( self.exfrom.get())
        i1=np.int( self.exto.get())	
        file0=np.int(np.floor(i0/self.nt))
        file1=np.int(np.floor(i1/self.nt))
        i0rel=i0-file0*self.nt
        i1rel=i1-file0*self.nt
        for inr,nni in enumerate(self.nn):    
        	print('loading processor files for ts at '+str(self.x[nni])+' '+str(str(self.y[nni])))
        	ii=np.where(self.node_proc_map[:,0]==nni)[0][0]			
        	procnr=self.node_proc_map[ii,1]
        	inde_rel=self.node_proc_map[ii,2]
        	self.procfiles=glob.glob(self.combinedDir+'schout_{:04d}_'+'*.nc'.format(procnr))
        	print(self.procfiles)
        	print(procnr)        	
        	nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.procfiles]
        	self.procfiles=np.asarray(self.procfiles)[np.argsort(nrs)][file0:file1+1]		
        	self.tsdsi=xr.open_mfdataset(self.procfiles)
        	
        	if  (self.dsi[self.varname].i23d == 1) & (self.dsi[self.varname].ivs == 1):
        	    tsi=self.tsdsi[self.varname][i0rel:i1rel,inde_rel]
        	elif  (self.dsi[self.varname].i23d == 2) & (self.dsi[self.varname].ivs == 1):
        	    tsi=self.tsdsi[self.varname][i0rel:i1rel,inde_rel,self.nn,self.lvl]			
        	elif  (self.dsi[self.varname].i23d == 1) & (self.dsi[self.varname].ivs == 2):
        	    tsi=self.tsdsi[self.varname][i0rel:i1rel,inde_rel,:]
        	elif  (self.dsi[self.varname].i23d == 2) & (self.dsi[self.varname].ivs == 2):
        	    tsi=self.tsdsi[self.varname][i0rel:i1rel,inde_rel,self.lvl,:]
        	#self.t=self.nclv['time'][i0:i1]
        	self.ts[inr,:]=tsi
        	self.t=self.tsdsi[self.varname][i0rel:i1rel,inde_rel,self.lvl,:]		

#	if self.shape==(self.nt,self.nnodes,self.nz):
#    self.nclv[self.varname][:,self.nn,self.lvl]
#    self.ts=self.nclv[self.varname][i0:i1,self.nn,self.lvl]
#elif self.shape==(self.nt,self.nnodes):
#    self.ts=self.nclv[self.varname][i0:i1,self.nn]
#elif self.shape==(self.nt,self.nnodes,2):
#    if self.npt==1:
#        self.ts=self.nclv[self.varname][i0:i1,self.nn,:].reshape(i1-i0,1,2)
#    else:
#        self.ts=self.nclv[self.varname][i0:i1,self.nn,:]
#elif self.shape==(self.nt,self.nnodes,self.nz,2):
#    if self.npt==1:
#        self.ts=self.nclv[self.varname][i0:i1,self.nn,self.lvl,:].reshape(i1-i0,1,2)
#    else:
#        self.ts=self.nclv[self.varname][i0:i1,self.nn,self.lvl,:]
            
        fig2=plt.figure(2)
        fig2.clf()
        if self.ivs==1:
            
            if self.CheckEval.get()!=0:
                expr= self.evalex.get()
                expr=expr[expr.index('=')+1:].replace('x','self.ts').replace('A','self.A').replace('dt','self.dt')
                self.ts=eval(expr)  
            
            plt.plot(self.t/86400,self.ts)
            plt.xlabel('time')
            plt.ylabel(self.varname)
            plt.grid()
            plt.legend(['P'+str(i) for i in range(self.npt)],loc='upper center',bbox_to_anchor=(0.5, 1.02),ncol=6)
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.plot(self.t/86400,self.ts[:,:,iplt-1])
                plt.tick_params(axis='x',labelbottom='off')
                plt.ylabel(self.varname + comps[iplt-1])
                plt.grid()
                if iplt==1:
                    plt.legend(['P'+str(i) for i in range(self.npt)],loc='upper center',bbox_to_anchor=(0.5, 1.3),ncol=6)


            if self.CheckEval.get()!=0:
                    expr= self.evalex.get()
                    expr=expr[expr.index('=')+1:].replace('x','ts').replace('A','self.A').replace('dt','self.dt')
                    self.nodevalues=eval(expr)    


            plt.subplot(3,1,3)
            plt.plot(self.t/86400,np.sqrt(self.ts[:,:,0]**2+self.ts[:,:,1]**2))
            plt.xlabel('time')
            plt.ylabel(self.varname + comps[2])
            plt.grid()
        
        plt.tight_layout()
        self.update_plots()
        print("done extracting time series")                                     
        

    def profiles(self,coords=None):
        """"
        Extract profiles at nextneighbours to clicked coordinates
        """
        self.extract=self.profiles
        if coords==None:
            self.ask_coordinates()
                        
        if self.shape==(self.nt,self.nnodes,self.nz):
            ps=self.ncv[self.varname][self.ti,self.nn,:]
            
        elif self.shape==(self.nt,self.nnodes,self.nz,2):
            ps=self.ncv[self.varname][self.ti,self.nn,:,:]
        else:
            print("variable has no depth associated")
            return
            
        zs=self.ncv['zcor'][self.ti,self.nn,:]
        
        if self.npt>1:
            ps=ps.swapaxes(0,1)
            zs=zs.swapaxes(0,1)
            
        fig2=plt.figure(2)
        fig2.clf()

        if self.ivs==1:
            plt.title(str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))))
            plt.plot(ps,zs)
            plt.ylabel('depth / m')
            plt.xlabel(self.varname)
            plt.legend(['P'+str(i) for i in range(self.npt)])
            plt.grid()
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(1,3,iplt)
                plt.plot(ps[:,:,iplt-1],zs)
                plt.grid()
                plt.xlabel(self.varname + comps[iplt-1])
                if iplt==1:
                    plt.title(str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))))
                    plt.legend(['P'+str(i) for i in range(self.npt)])
                    plt.ylabel('depth / m')
                else:
                    plt.tick_params(axis='y',labelleft='off')
            plt.subplot(1,3,3)
            plt.plot(np.sqrt(ps[:,:,0]**2+ps[:,:,1]**2),zs)
            plt.xlabel(self.varname + comps[2])
            plt.grid()
            plt.tick_params(axis='y',labelleft='off')
            plt.tight_layout()
            
        self.zminfield.delete(0, 'end')
        self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
        self.update_plots()

    def hovmoeller(self,coords=None):
        
        self.extract=self.hovmoeller
        if coords==None:
            self.ask_coordinates(n=1)

        print('extracting hovmoeller for ' + self.varname + ' at coordinats: ' + str(self.coords))
        i0,i1=int(self.exfrom.get()),int(self.exto.get())

        self.zcor=np.squeeze(self.nclv['zcor'][i0:i1,self.nn,:])
        self.t=self.nclv['time'][i0:i1]
        
        if self.shape==(self.nt,self.nnodes,self.nz):
            self.ts=np.squeeze(self.nclv[self.varname][i0:i1,self.nn,:])
        elif self.shape==(self.nt,self.nnodes) or self.shape==(self.nt,self.nnodes,2):
            print("variablae has no depths")
            return
        elif self.shape==(self.nt,self.nnodes,self.nz,2):
            self.ts=np.squeeze(self.nclv[self.varname][i0:i1,self.nn,:])

        fig2=plt.figure(2)
        fig2.clf()
        if self.ivs==1:
            plt.pcolor(np.tile(self.t/86400,(self.nz,1)).transpose(),self.zcor,self.ts)
            plt.ylabel('depth')
            plt.title(self.varname)
        else:
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.pcolor(np.tile(self.t/86400,(self.nz,1)).transpose(),self.zcor,np.squeeze(self.ts[:,:,iplt-1]))
                plt.ylabel('depth')
                plt.title(self.varname)
                plt.colorbar()
                plt.title(self.varname + comps[iplt-1])
                plt.tick_params(axis='x',labelbottom='off')
                plt.set_cmap(self.cmap.get())
            plt.subplot(3,1,3)
            plt.pcolor(np.tile(self.t/86400,(self.nz,1)).transpose(),self.zcor,np.sum(np.sqrt(self.ts**2),axis=-1))
            plt.title(self.varname + comps[2])  
            
        plt.colorbar()
        plt.xlabel('time')
        plt.tight_layout()
        plt.set_cmap(self.cmap.get())
        
        self.zminfield.delete(0, 'end')
        self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0]))
        self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
        self.update_plots()
        

    def transect_callback(self,coords=None):

        self.extract=self.transect_callback #'transect'
        if coords==None:
            self.ask_coordinates()
            # einmalig
            if self.minterp!='inv_dist10' and self.minterp!='nn':
                self.inds=self.faces[self.parents].flatten()  # shape 2,3
                self.indrange=np.arange(self.npt*3)
                self.ibtm=self.ncv['node_bottom_index'][self.inds]-1 
                self.zrange=np.tile(np.arange(58),(3,self.npt,1)).swapaxes(0,2)
                zbtm=-self.ncv['depth'][self.inds]
                self.zbtmi=zbtm.reshape((self.npt,3))
                self.zinterp=np.zeros((58,self.npt,3))
                self.varinterp=np.zeros((58,self.npt,3))
                self.w_z_pt_nde= self.wtrans.swapaxes(0,2)


        # new interp nicht ganz correct            

        #interp in between
        if self.minterp=='nn':  # nearest neighbour interpolation
            if self.shape==(self.nt,self.nnodes,self.nz):
                ps=self.ncv[self.varname][self.ti,self.nn,:].swapaxes(0,1)
            elif self.shape==(self.nt,self.nnodes,self.nz,2):
                ps=self.ncv[self.varname][self.ti,self.nn,:,:].swapaxes(0,1)
            else:
                print("variable has no depth associated")
                return
            zs=self.ncv['zcor'][self.ti,self.nn,:].swapaxes(0,1)
            #xs=np.tile(range(self.npt),(self.nz,1)).transpose() # transponiert zu meinem ansatz
            #print(xs.shape)
            #print(zs.shape)
            #print(ps.shape)
            #plt.pcolor(xs.swapaxes(0,1),zs,ps)
        elif self.minterp=='bary_sigma': # barycentric interpolation along sigma levels
            if self.shape==(self.nt,self.nnodes,self.nz):  # reshaping is wrong maybe strange values near mask -> fill bottom with last valuse
                #ps=np.sum(self.ncv[self.varname][self.ti,self.faces[self.parents].flatten(),:].reshape(3,self.npt,58)*self.wtrans,axis=0).swapaxes(0,1)
                
                # fill lower levels
                pdata=self.ncv[self.varname][self.ti,self.faces[self.parents].flatten(),:]                
                zdata=self.ncv['zcor'][self.ti,self.faces[self.parents].flatten(),:]                
                for ii,ind in enumerate(self.faces[self.parents].flatten()):
                    pdata[ii,:self.ncv['node_bottom_index'][ind]-2]=pdata[ii,self.ncv['node_bottom_index'][ind]-1]
                    zdata[ii,:self.ncv['node_bottom_index'][ind]-2]=zdata[ii,self.ncv['node_bottom_index'][ind]-1]
                
                ps=np.sum(pdata.reshape(3,self.npt,58)*self.wtrans,axis=0).swapaxes(0,1)
                
            elif self.shape==(self.nt,self.nnodes,self.nz,2):
                #ps=np.sum(self.ncv[self.varname][self.ti,self.faces[self.parents].flatten(),:,:].reshape(3,self.npt,58,2)*self.wtransvec,axis=0).swapaxes(0,1)
                pdata=self.ncv[self.varname][self.ti,self.faces[self.parents].flatten(),:,:]                
                zdata=self.ncv[self.varname][self.ti,self.faces[self.parents].flatten(),:]                
                # fill lower levels
                for ii,ind in enumerate(self.faces[self.parents].flatten()):
                    pdata[ii,:self.ncv['node_bottom_index'][ind]-2,:]=pdata[ii,self.ncv['node_bottom_index'][ind]-1,:]
                    zdata[ii,:self.ncv['node_bottom_index'][ind]-2]=zdata[ii,self.ncv['node_bottom_index'][ind]-1]

            else:
                print("variable has no depth associated")
                return
            #zs=np.sum(self.ncv['zcor'][self.ti,self.faces[self.parents].flatten(),:].reshape(3,self.npt,58)*self.wtrans,axis=0).swapaxes(0,1)
            zs=np.sum(zdata.reshape(3,self.npt,58)*self.wtrans,axis=0).swapaxes(0,1)
                
            
        elif self.minterp=='inv_dist10':
            z=self.ncv['zcor'][self.ti] # fill with bottom values 
            s=self.ncv[self.varname][self.ti]
            zs=np.ones((len(self.dist[:,0]),self.nz))
            if self.ivs==1:
                ps=np.ma.masked_equal(np.ones((len(self.dist[:,0]),self.nz)),0.0)
                for n in range(self.nz-1,-1,-1):
                    ps[:,n] = s[self.inds,n].mean(axis=1)
                    zs[:,n] = z[self.inds,n].mean(axis=1)
            else:
                ps=np.ma.masked_equal(np.ones((len(self.dist[:,0]),self.nz,self.ivs)),0.0)
                for n in range(self.nz-1,-1,-1):
                    ps[:,n,:] = s[self.inds,n,:].mean(axis=1)
                    zs[:,n] = z[self.inds,n].mean(axis=1)

            #ps=np.sum(self.ncv[self.varname][ti,self.faces[self.parents].flatten(),:,:].reshape(3,self.npt,58,2)*self.wtransvec,axis=0).swapaxes(0,1)
            
        else: # vert z plus barycentric interpolation

            variable=np.array((self.ncv[self.varname][self.ti,self.inds,:]))#
            z=np.array((self.ncv['zcor'][self.ti,self.inds,:]))#
            zsurf=z[self.indrange,-1]
            zsurfi=zsurf.reshape((self.npt,3))
            zi= + self.zbtmi + self.zrange*np.tile((zsurfi-self.zbtmi)/57,(58,1,1))
            zi2=np.sum(np.array(self.w_z_pt_nde*zi),axis=2)  # interpolation levels

            # interpolate z
            trinde=0
            for inde in range(self.npt*3):
                trinde=inde%3
                ipt=int(np.floor(inde/3))
                trinde*=inde%3!=0
                self.zinterp[:,ipt,trinde]=(scipy.interpolate.griddata(z[inde,self.ibtm[inde]:], z[inde,self.ibtm[inde]:], zi2[:,ipt] , method='linear', fill_value=np.nan, rescale=False)   )
                self.varinterp[:,ipt,trinde]=(scipy.interpolate.griddata(z[inde,self.ibtm[inde]:], variable[inde,self.ibtm[inde]:], zi2[:,ipt] , method='linear', fill_value=np.nan, rescale=False)   )
                trinde+=1

                # re weight nan
                inan=np.isnan(self.zinterp) # einmalig
                w1=self.w_z_pt_nde*~inan
                w2=w1/np.tile(w1.sum(axis=2),(3,1,1)).swapaxes(0,1).swapaxes(1,2)

                self.zinterp[inan]=0  # check here
                self.varinterp[inan]=0
                zs=np.sum(self.zinterp*w2,axis=2) #-zi2 #interpolate horz
                ps=np.sum(self.varinterp*w2,axis=2) #-zi2 #interpolate horz


        
        
        if self.minterp=='inv_dist10':
            xs=np.tile(range(self.npt),(self.nz,1)).transpose() # transponiert zu meinem ansatz
        else:
            xs=np.tile(range(self.npt),(self.nz,1))

        print('variable shape')
        print(xs.shape)
        print(zs.shape)
        print(ps.shape)

            
        ylim=((zs.min(),5))
        fig2=plt.figure(2)
        fig2.clf()

        if self.ivs==1: # scalar
            #plt.pcolor(xs,zs,ps,shading=['flat','faceted'][self.meshVar.get()]) # shading error recently ?
            plt.pcolor(xs,zs,ps)	            	
            #plt.plot(xs.T,zs.T,'k--')
            #plt.plot(xs.T[8,:],zs.T[8,:],'k--')
            #for i in range(xs.shape[0]):
            #    plt.text(xs.T[0,i],zs.T[0,i],str(i))
            plt.ylabel('depth / m')
            plt.title(str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))))
            ch=plt.colorbar()
            ch.set_label(self.varname)
        else: # vector
            comps=[' - u', '- v ', '- abs' ]
            for iplt in range(1,3):
                plt.subplot(3,1,iplt)
                plt.pcolor(xs,zs,ps[:,:,iplt-1],shading=['flat','faceted'][self.meshVar.get()])
                plt.ylabel('depth / m')
                plt.tick_params(axis='x',labelbottom='off')
                plt.set_cmap(self.cmap.get())
                ch=plt.colorbar()
                ch.set_label(self.varname + comps[iplt-1])
                plt.gca().set_ylim(ylim)
                plt.clim(self.clim) 
                
                if iplt==1:
                    plt.title(str(self.reftime + dt.timedelta(seconds=np.int(self.ncv['time'][self.ti]))))
            plt.subplot(3,1,3)
            plt.pcolor(xs,zs,np.sqrt(ps[:,:,0]**2+ps[:,:,1]**2),shading=['flat','faceted'][self.meshVar.get()])
            ch=plt.colorbar()
            ch.set_label(self.varname + comps[-1])

        plt.gca().set_ylim(ylim)
        plt.clim(self.clim) 
        plt.xlabel('transect length [points]')   
        plt.set_cmap(self.cmap.get())
        plt.tight_layout()
        self.zminfield.delete(0, 'end'),self.zmaxfield.delete(0, 'end')
        self.zminfield.insert(8,str(plt.gca().get_ylim()[0])),self.zmaxfield.insert(8,str(plt.gca().get_ylim()[1]))
        self.update_plots()
            
        
    def client_exit(self):
        plt.close('all')
        self.master.destroy()
        exit()

    # navigate time steps    
    def firstts(self):
        self.stack.set(self.stacks[0])
        self.ti_tk.set(0)
    def prevstep(self):
        if self.ti>=0:
            self.ti_tk.set(self.ti-1)
        elif self.stack.get()-1 in self.stacks:
            self.stack.set(self.stack.get()-1)
    def nextstep(self):
        if self.ti<self.nt-1:
            self.ti_tk.set(self.ti+1)
        elif self.stack.get()+1 in self.stacks:
            self.stack.set(self.stack.get()+1)
        else:   
            self.stack.set(self.stacks[0])
    def lastts(self):
        self.stack.set(self.stacks[self.nstacks-1])
        self.ti_tk.set(self.nt-1)

    def updateaxlim(self,*args):
        plt.figure(1)
        axes = plt.gca()
        xmin,xmax=self.xminfield.get(),self.xmaxfield.get()
        ymin,ymax=self.yminfield.get(),self.ymaxfield.get()
        if (len(xmax)*len(ymax)*len(ymin)*len(ymax))>0:
            axes.set_xlim([np.double(xmin),np.double(xmax)])
            axes.set_ylim([np.double(ymin),np.double(ymax)])
            self.update_plots()
            
    def updatezlim(self,*args):
        if 2 in plt.get_fignums() and (self.extract!=self.timeseries):
            zmin,zmax=self.zminfield.get(),self.zmaxfield.get()
            if (len(zmax)*len(zmin))>0 : 
                cf=plt.figure(2)
                if self.ivs==1:
                    axes = plt.gca()
                    cf.get_axes()
                    axes.set_ylim([np.double(zmin),np.double(zmax)])
                else:
                    if self.extract==self.profiles:
                        m,n=1,3
                    else:
                        m,n=3,1
                    for i in range(1,4):
                        plt.subplot(m,n,i)    
                        axes = plt.gca()
                        cf.get_axes()
                        axes.set_ylim([np.double(zmin),np.double(zmax)])
                self.update_plots()
                
# launch gui
root = tk.Tk()
root.geometry("400x580")
root.grid_rowconfigure(12, minsize=100)  
root.grid_columnconfigure(4, minsize=100)  
app= Window(root)
root.title('schout_view')
root.mainloop()


#app.variable.set('elev')
#app.quivVar.set(1)
#app.variable.set('wind_speed')