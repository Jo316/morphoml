#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
import sys 
from keyvars import ufiles_path
sys.path.append(ufiles_path)

import uvars
import uviz
import uprocessing as up 
import utransform as ut 
import uerrortab as etab 
import umodelling as uml 
import os 
from glob import  glob
from pyspatialml import  Raster
import numpy as np 
import pandas as pd 
from pprint import  pprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor 
from xgboost import  XGBRFRegressor, XGBRegressor
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,BayesianRidge
from sklearn.isotonic import IsotonicRegression 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor,StackingRegressor
from sklearn.model_selection import cross_val_score, KFold,cross_validate
import pickle 

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# In[2]:


fcols = ['cop', 'edem', 'pband', 's1', 'tdemx', 'wc', 'wc_ffth',
       'wc_gau', 'wc_sobelm', 'wc_sobelh', 'wc_sobelv']

tcol = 'tdemx'
psize = 256
num_cpus = 10
seed = 1302


# In[3]:


parqts = uvars.parqts
df = pd.read_parquet(parqts[1])


# In[4]:


df.head()


# In[5]:


models = {
    "LGBM": LGBMRegressor(num_threads=num_cpus,seed =seed,verbosity=-1),
    "CatBoost": CatBoostRegressor(thread_count=num_cpus, verbose=0),
   
    "RandomForest": RandomForestRegressor(n_jobs=num_cpus),
    "LinearRegression": LinearRegression(n_jobs=num_cpus),

    "Ridge": Ridge(),
    "Lasso": Lasso(),

    "BayesianRidge": BayesianRidge(),
    "IsotonicRegression": IsotonicRegression(out_of_bounds="clip"),
    
    "MLP": MLPRegressor(early_stopping=True),

    "XGBRF": XGBRFRegressor(n_jobs=num_cpus),#, verbosity=0),
    "XGB": XGBRegressor(n_jobs=num_cpus),#, verbosity=0),
    
}


# In[11]:


models_e = [
    ("LGBM", LGBMRegressor(verbosity=-1)),
    ("CatBoost", CatBoostRegressor(verbose=0)),
   
    ("RandomForest", RandomForestRegressor()),
    ("LinearRegression", LinearRegression()),

    ("Ridge", Ridge()),
    ("Lasso", Lasso()),

    ("BayesianRidge", BayesianRidge()),
    ("IsotonicRegression",IsotonicRegression(out_of_bounds="clip")),
    
    ("MLP", MLPRegressor(early_stopping=True))]


# In[12]:


rf = RandomForestRegressor(n_jobs=num_cpus) # best model not necessarily rf
voting_regr = VotingRegressor(estimators=models_e, n_jobs=num_cpus)
stacking_regr = StackingRegressor(estimators=models_e, final_estimator=rf)#, n_jobs=num_cpus)


# In[8]:


dir_v1 = os.path.join(uvars.idata_tilepath, 'aexp_v1')
os.makedirs(dir_v1, exist_ok=True)


# In[9]:


# can you fit one variable in xgboost as feature


# In[10]:


for j,fcol in enumerate(fcols):
    print('****'*100)
    error_list = []
    #if j > 0 : break
    dir_fvar = os.path.join(dir_v1, fcol); os.makedirs(dir_fvar, exist_ok=True)
    trainx, validx, trainy,validy = train_test_split(df[fcol].values, df[tcol].values, 
                                                    test_size=0.15, random_state=seed)

    


    for i, (name, model) in enumerate(models.items()):
        print(f'Fitting {j} {fcol} {i} {name}')
        #if i > 4: break
        print(i,':',name)
        modelname = f"{name}_default_{fcol}_{len(trainx)}_{str(1)}"
        model_pkl = os.path.join(dir_fvar, f"{modelname}.pkl")
        model_csv = os.path.join(dir_fvar, f"{modelname}.csv")
        
        #print(model_pkl)
        ##### function 
        try:
            ee = uml.fit_model(model, trainx, trainy, validx, validy, model_pkl,fcol, model_csv)
            error_list.append(ee)
            #tabulate(ee)
        except:
            pass 
        
    print('Fitting Votting Regression ')
    try:
        model_csv_vot = os.path.join(dir_fvar, f"VotingRegressor_default_{fcol}_{len(trainx)}_{j}.csv")
        model_pkl_vot = os.path.join(dir_fvar, f"VotingRegressor_default_{fcol}_{len(trainx)}_{j}.pkl")
        ee = uml.fit_model(voting_regr, trainx, trainy, validx, validy, model_pkl_vot,fcol, model_csv_vot)
        error_list.append(ee)
    except:
        pass 
    
    try:
        print('Fitting Stacking Regression ') # porblems with stacking 
        model_csv_stk = os.path.join(dir_fvar, f"StackingRegressor_default_{fcol}_{len(trainx)}_{j}.csv")
        model_pkl_stk = os.path.join(dir_fvar, f"StackingRegressor_default_{fcol}_{len(trainx)}_{j}.pkl")

        ee = uml.fit_model(stacking_regr, trainx, trainy, validx, validy, model_pkl_stk,fcol, model_csv_stk)
        error_list.append(ee)
    except:
        pass

    E = pd.concat(error_list, ignore_index=True)
    Emodel_csv = os.path.join(dir_fvar, f"{tcol.upper()}_models_default_{i}_{j}.csv")
    E.to_csv(Emodel_csv, index=False)
    pprint(E)
    print('****'*100)
            
    

    
    

    


# In[ ]:




