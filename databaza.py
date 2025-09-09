import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pycaret  import regression
from pycaret.regression import create_model,tune_model,plot_model,predict_model
from sklearn.ensemble import VotingRegressor #Ensembly
from transformers import pipeline #sentiment analysis
from sklearn.model_selection import cross_val_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

BMW=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/BMW_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')
KTM=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/KTM_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')
RE=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/Royal_Enfield_Standard_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')
SUZ=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/Suzuki_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')
YAM=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/Yamaha_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')
DUC=pd.read_csv('/kaggle/input/comprehensive-motorcycles-dataset/ducatti_bike.csv',encoding= 'unicode_escape',on_bad_lines='skip')

Brands=[BMW,KTM,RE,SUZ,YAM,DUC]
for i in Brands:
    print(i.shape,i.columns)

BMW['brand']='BMW'
KTM['brand']='KTM'
RE['brand']='RE'
SUZ['brand']='SUZ'
YAM['brand']='YAM'
DUC['brand']='DUC'

BMW.rename(columns={'mileage':'DIST','price':'PRICE','Bike':'MODEL','Types and Used Time':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)
KTM.rename(columns={'mileage':'DIST','price':'PRICE','Bike':'MODEL','Types and Used Time':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)
RE.rename(columns={'mileage':'DIST','price':'PRICE','bike':'MODEL','Types ':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)
SUZ.rename(columns={'mileage':'DIST','price':'PRICE','BIke name':'MODEL','Types and Used Time':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)
YAM.rename(columns={'mileage':'DIST','price':'PRICE','Bike name':'MODEL','Types and Used  Time':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)
DUC.rename(columns={'mileage':'DIST','price':'PRICE','Bike name ':'MODEL','Time of USed':'YEAR','description':'DESC','brand':'BRAND'},inplace=True)

VDB=pd.concat([BMW,KTM,RE,SUZ,YAM,DUC],axis=0,ignore_index=True)

VDB.head()

VDB.isnull().sum()

from transformers import pipeline
classifier=pipeline('sentiment-analysis',max_length=512,truncation=True)
Sentiment_score=[]
for i in range(len(VDB.DESC)):
    if(isinstance(VDB.DESC[i],str)):
        Sentiment_score.append((classifier(VDB.DESC[i]))[0]['score'])
    else:
        Sentiment_score.append(np.nan)

VDB['DESC_SENTIMENT']=Sentiment_score
VDB=VDB.drop(columns=['DESC'],axis=1)
VDB.head()

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
VDB["DESC_SENTIMENT"] = imp.fit_transform(VDB[["DESC_SENTIMENT"]]).ravel()
VDB.head()

VDB = VDB.dropna(axis=0, subset=['DIST'])

VDB.DIST=VDB.DIST.str.rstrip(' miles')

VDB.PRICE=VDB.PRICE.str.lstrip('$')
VDB.head()

from dateutil.parser import parse
VDB.reset_index(inplace = True, drop = True)
Year=[]
for i in range(len(VDB.YEAR)):
    Year.append(2023-(parse(VDB.YEAR[i],fuzzy=True).year))
VDB.YEAR=Year
VDB.rename(columns={'YEAR':'YEARS_USED'},inplace=True)
VDB.head()

VDB=VDB[ VDB['PRICE' ].str.contains( 'No Price Listed' )==False ]
VDB=VDB[ VDB['PRICE' ].str.contains( 'Call For Price' )==False ]

VDB.reset_index(inplace = True, drop = True)

VDB['DIST'] = VDB['DIST'].str.replace(',', '').astype(float)
VDB['PRICE'] = VDB['PRICE'].str.replace(',', '').astype(float)
VDB.head()

from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(enc.fit_transform(VDB[['BRAND']]))

OH_cols_train.index = VDB.index

num_X_train = VDB.drop(columns=['BRAND'], axis=1)

VDB= pd.concat([num_X_train, OH_cols_train], axis=1)


VDB.columns = VDB.columns.astype(str)

VDB.head()

VDB.MODEL.nunique()

freq_map=VDB.MODEL.value_counts().to_dict()
VDB.MODEL=VDB.MODEL.map(freq_map)
VDB.head()

X=VDB.drop(columns=['PRICE'],axis=1)
Y=VDB.PRICE

regression.setup(X,target=Y ,session_id=42)
regression.compare_models()

cb = create_model('catboost')
et = create_model('et')

tune_cb=tune_model(cb)
tune_et=tune_model(et)

ensemble_model = VotingRegressor(estimators=[('catboost',cb ),('et', et)] ,weights=[1,1])

scores = cross_val_score(ensemble_model, X, Y, cv=5)
print(scores)

ensemble_model.fit(X,Y)

preds = predict_model(ensemble_model)
preds

plot_model(ensemble_model)

plot_model(ensemble_model, plot = 'error')
