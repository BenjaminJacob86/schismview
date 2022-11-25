self.ncs[self.vardict['depth']]['depth'][self.total_time_index]


## add computational varname To list
varname1='sedDepositionalFlux'
varname2='sedErosionalFlux'
varname='sedNetDeposition'
self.varlist=list(self.varlist)+[varname,]
self.vardict[varname]=varname
self.ncs[varname]={varname:self.ncs[self.vardict[varname1]][varname1]-self.ncs[self.vardict[varname2]][varname2]}
self.varlist=np.sort(self.varlist)
self.combo['values']=np.sort(self.combo['values']+(varname,))





# load files    
if ncdirsel==None:
	print("navigate into schsim run directory (containing param.nml)")
	self.runDir=filedialog.askdirectory(title='enter run directory direcory')+'/'
	nnodes=np.int(np.loadtxt(self.runDir+'hgrid.ll',skiprows=1,max_rows=1)[1])
	m=np.loadtxt(self.runDir+'hgrid.ll',skiprows=2,max_rows=nnodes)
	self.lon,self.lat=m[:,1],m[:,2]
	self.ll_nn_tree = cKDTree([[self.lon[i],self.lat[i]] for i in range(len(self.lon))])
	print("navigate into schout_*.nc directory")
	self.combinedDir=filedialog.askdirectory(title='enter schout_*.nc direcory')+'/'
else:
	self.combinedDir=ncdirsel
	
# new i/o files	
if len(glob.glob(self.combinedDir+'out2d_*.nc'))>0:
	print('found per variable netcdf output format')
	self.oldio=False
elif len(self.combinedDir+'schout_*.nc')>0:
	print('found schout.nc output format')
	self.oldio=True


self.files=[] 		
if self.oldio:
	for iorder in range(6): # check for schout_nc files until 99999
		self.files+=glob.glob(self.combinedDir+'schout_'+'?'*iorder+'.nc')
	nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.files]
	self.files=list(np.asarray(self.files)[np.argsort(nrs)])
	nrs=list(np.asarray(nrs)[np.argsort(nrs)])
	self.nstacks=len(self.files)
	print('found ' +str(self.nstacks) +' stack(s)')
	self.stack0=np.int(nrs[0])
	
	# initialize extracion along files # check ofr future if better performance wih xarray
	self.ncs={'schout':[]}
	self.ncs['schout']=xr.concat([ xr.open_dataset(self.combinedDir+'schout_'+str(nr)+'.nc').chunk() for nr in nrs],dim='time')
	self.ncv=self.ncs['schout'].variables
	try:
		self.ncs['schout']=xr.concat([ xr.open_dataset(self.combinedDir+'schout_'+str(nr)+'.nc').chunk() for nr in nrs],dim='time')
	except:
		print("error loading via MFDataset - time series and hovmoeller diagrams wont work")
		pass		
		
		
	self.vardict={} # variable to nc dict relations
	
   # load variable list from netcdf #########################################            
	exclude=['time','SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes', 'SCHISM_hgrid_node_x',
 'SCHISM_hgrid_node_y', 'bottom_index_node', 'SCHISM_hgrid_face_x', 'SCHISM_hgrid_face_y', 
 'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y', 'edge_bottom_index',
 'sigma', 'dry_value_flag', 'coordinate_system_flag', 'minimum_depth', 'sigma_h_c', 'sigma_theta_b', 
 'sigma_theta_f', 'sigma_maxdepth', 'Cs', 'wetdry_node','wetdry_elem', 'wetdry_side'] # exclude for plot selection
	vector_vars=[] # stack components for convenience	  
	self.vardict={} # variable to nc dict relations

	for vari in self.ncv:
		if vari not in exclude:
			if  self.ncv[vari].shape[-1]==2:
				vector_vars.append(vari)		
				self.vardict[vari]=vari			
				self.ncs[vari] ={vari: xr.concat([self.ncs['schout'][vari].sel(two=0),self.ncs['schout'][vari].sel(two=1)], dim='ivs')}
			else:
				self.vardict[vari]='schout'		
	self.varlist=list(self.vardict.keys())
	self.filetag='schout'			
	self.bindexname='node_bottom_index'
	self.zcorname='zcor'			
	self.dryvarname='wetdry_elem'
	self.hvelname='hvel'
	self.vertvelname='vertical_velocity'			
	# work around to map old velocity as new velocity formatted

	strdte=[np.float(digit) for digit in self.ncs[self.filetag]['time'].attrs['base_date'].split()]
	self.reftime=dt.datetime.strptime('{:04.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:{:02.0f}'.format(strdte[0],strdte[1],strdte[2],strdte[3],strdte[3],0),'%Y-%m-%d %H:%M:%S')
	
else: # new io
	self.hvelname='horizontalVel'
	self.filetag='out2d'
	self.bindexname='bottom_index_node'
	self.zcorname='zCoordinates'
	self.dryvarname='dryFlagElement'		
	self.vertvelname='verticalVelocity'			
	for iorder in range(8): # check for schout_nc files until 99999
		self.files+=glob.glob(self.combinedDir+'out2d_'+'?'*iorder+'.nc')
	nrs=[int(file[file.rfind('_')+1:file.index('.nc')]) for file in self.files]
	self.files=list(np.asarray(self.files)[np.argsort(nrs)])
	nrs=list(np.asarray(nrs)[np.argsort(nrs)])
	self.nstacks=len(self.files)
	self.stack0=np.int(nrs[0])
	print('found ' +str(self.nstacks) +' stack(s)')

	
	# file access        
	# vars # problem sediment
	varfiles=[file[file.rindex('/')+1:file.rindex('_')] for file in glob.glob(self.combinedDir+'*_'+str(nrs[0])+'.nc') ]
	self.ncs=dict.fromkeys(varfiles)
	try:
		for var in varfiles:
			self.ncs[var]=xr.concat([ xr.open_dataset(self.combinedDir+var+'_'+str(nr)+'.nc').chunk() for nr in nrs],dim='time')
	except:	
		nrs=list(np.asarray(nrs)[np.argsort(nrs)])[:-1]
		self.nstacks=len(self.files)-1
		for var in varfiles:
			self.ncs[var]=xr.concat([ xr.open_dataset(self.combinedDir+var+'_'+str(nr)+'.nc').chunk() for nr in nrs],dim='time')		


   # load variable list from netcdf #########################################            
	exclude=['time','SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes', 'SCHISM_hgrid_node_x',
 'SCHISM_hgrid_node_y', 'bottom_index_node', 'SCHISM_hgrid_face_x', 'SCHISM_hgrid_face_y', 
 'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y', 'edge_bottom_index',
 'sigma', 'dry_value_flag', 'coordinate_system_flag', 'minimum_depth', 'sigma_h_c', 'sigma_theta_b', 
 'sigma_theta_f', 'sigma_maxdepth', 'Cs', 'dryFlagElement'] # exclude for plot selection
	vector_vars=[] # stack components for convenience	  
	self.vardict={} # variable to nc dict relations
	for nci_key in self.ncs.keys():
		for vari in self.ncs[nci_key].keys():
			if vari not in exclude:
				self.vardict[vari]=nci_key	
			if vari[-1] =='Y': 
				vector_vars.append(vari[:-1])

	self.varlist=list(self.vardict.keys())

	for vari_vec in vector_vars:			
		varX=vari_vec+'X'	  
		varY=vari_vec+'Y'	  
		self.varlist+=[vari_vec]
		self.vardict[vari_vec]=vari_vec
		self.ncs[vari_vec] ={vari_vec: xr.concat([self.ncs[self.vardict[varX]][varX], self.ncs[self.vardict[varY]][varY]], dim='ivs')}