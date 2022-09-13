# -*- coding: utf-8 -*-
"""
Demo file of Newt_ALM for solving SLOPE model

"""
import sys
sys.path.append( './WeightedK/nal' )
sys.path.append( './WeightedK/admm' )
sys.path.append( './BioNUS' )
sys.path.append( './Solvers' )
sys.path.append( './mexfun' )

import os
import time
import numpy as np
from   pathlib import Path
import scipy.io as scio
import scipy
import WeightedK.nal.Newt_ALM
from WeightedK import Newt_ALM
import scipy.sparse




# load path

HOME        = os.getcwd()
HOME_runexp = HOME + '\\' + 'RESULTS'

# define name of data_file  - use dict
fname = {
    1 : 'E2006.train',
    2 : 'log1p.E2006.train',
    3 : 'E2006.test',
    4 : 'log1p.E2006.test',
    5 : 'pyrim-scaled-expanded5',
    6 : 'triazines-scaled-expanded4',
    7 : 'abalone_scale_expanded7',
    8 : 'bodyfat_scale_expanded7',
    9 : 'housing_scale_expanded7',
    10 : 'mpg_scale_expanded7',
    11 : 'space_ga_scale_expanded9',
    
    21 : 'DLBCL_H',
    22 : 'lung_H1',
    23 : 'NervousSystem',
    24 : 'ovarian_P',
    25 : 'DLBCL_N',
    26 : 'DLBCL_S',
    27 : 'lung_H2',
    28 : 'lung_M',
    29 : 'lung_O',
    30 : 'ovarian_S'
    } 

rundate         = []
if not rundate:
    rundate      = time.strftime( "%d-%m-%Y" )
expdir           = HOME_runexp + '\\' + rundate + '\\' + 'BioNUS_WK'
saveyes          = 0
diaryyes         = 1
Path_expdir      = Path( expdir )
# if (saveyes or diaryyes) and ~Path_expdir.is_dir():
#     os.makedirs(expdir)
    
# if diaryyes:
#     diaryname = expdir + '\\' + 'run ' +  rundate + 'testBioNUS_SLOPE_BH.txt'


# input data: A b lambda
for i in [23]:
    if i < 20:
        datadir  = HOME + '\\' + 'UCIdata'
    elif i <= 30:
        datadir  = HOME + '\\' + 'BioNUS'
    probname     = datadir + '\\' + fname[i]
    print( 'Problem name: {}'.format(fname[i]) )
    dataFilePath = Path( probname + '.mat' )
    data         = scio.loadmat( dataFilePath ) # dict
    A            = data['A']
    b            = data['b']
    m,n          = A.shape
    preposs      = 1
    if preposs:
        D        = np.sqrt( sum(A*A,1) )
        A        = A[:,D>1e-12]
        norg     = n
        _,n      = A.shape
        if norg - n > 0:
            print('preprocess A: norg = %3.0d, n = %3.0d', norg, n)
        AAt      = A @ A.T # ATA
        R        = np.linalg.cholesky( AAt+1e-15*np.eye(m) )
        idx      = [k for ( k,val ) in enumerate( np.diag(R) ) if val<1e-8]
        ss       = set( range(m) ).difference( set(idx) )
        if len(ss) < m:
            A    = A[ss,:]
        morg     = m
        m,_      = A.shape
        if morg - m >0:
            print( 'morg = %3.0d, m = %3.0d', morg, m )
        b        = b[0:m]
    
    scale_data   = 2
    if scale_data:
        print( 'Data scale:',end =' ' )
    if scale_data == 1:
        D        = np.diag( np.maximum( 0,np.sqrt(sum(A*A,0)) ) )
        DA       = scipy.sparse.csc_matrix( D )
        A        = A @ DA
    elif scale_data == 2:
        normb    = np.linalg.norm( b ) 
        bscale   = max( 1,np.sqrt(np.sqrt(normb)) )
        b        = b/bscale
        A        = A/bscale
        print( 'scale data by {:.2e}'.format(bscale) )
    
    hR           = lambda x: A@x
    hRt          = lambda x: A.T@x 
    ''' tuning parameters  '''
    lambdamax    = np.linalg.norm(hRt(b),float('Inf'))
    tau_s        = 1e-3
    weight1      = tau_s*lambdamax
    weight2      = weight1/np.sqrt(n)
    lambda_lam   = weight1 + weight2 * np.arange(n-1,-1,-1).reshape(n,1)
    
    for stoprho in [6]:
        lambdaorg       = lambda_lam
        for crho in [1]:
            lambda_lam  = crho*lambdaorg
            for jj in [1]:
                init = np.zeros([n,1])
                stoptol = 10**( -stoprho )
                saveyes = 0
                eigsopt_issym = 1
                Rmap    = lambda x: A @ x
                Rtmap   = lambda x: A.T @ x
                RRtmap  = lambda x: Rmap( Rtmap(x) )
                '''这里求特征值'''
                AAT     = A@A.T
                Lipeig,Lipvector = np.linalg.eig( AAT )
                Lip     = max( Lipeig )
                print( '-----------------------' )
                print( 'Problem: [n= {}, m={}], [w1 = {:.2e} w2 = {:.2e}]'.format(n,m,float(lambda_lam[-1]),float(lambda_lam[-1])-float(lambda_lam[-2])) )
                print( '-----------------------' )
                class nalop:
                    pass
                nalop.Lip               = Lip
                nalop.stoptol           = stoptol
                nalop.runphaseI         = 1
                
                x0                      = np.zeros([n,1])
                xi0                     = np.zeros([m,1])
                u0                      = np.zeros([n,1]);
                obj,x,xi,u,info,runhist = Newt_ALM.Newt_ALM( A,b,n,lambda_lam,nalop,x0,xi0,u0 )
                Snal_res                = info
                del info,runhist
            #      if saveyes
            #     save([expdir,filesep,fname{i},'-',num2str(crho),'-',num2str(jj),'-',num2str(stoprho),'-Snal','.mat'],'Snal_res'); 
# if diaryyes:
#     diary off 
