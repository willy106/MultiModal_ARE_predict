import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def pre_tabular(df,targets,mode):

    df=df.dropna(axis=0, how='any')
    
    scaler = MinMaxScaler()
   


    # df = df[df['Hx no. (VGH)'] == '2754059-6']
    
    # df = df[df['Hx no. (VGH)'] != '2754059-6']

    if targets == '(ARE)':
        if mode=='test':
                df= df.drop([
                    'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS'], axis=1)
        else:
                df = df.drop([
                    'pattern ARE', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Brain stem', 'Cerebellum', 'Brain stem',
                    'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS','index'], axis=1)

    elif targets == 'pre CO  H':
            df = df.drop(['(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                'Brain stem', 'Cerebellum', 'Brain stem',
                'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
       'GKS'], axis=1)
    else:
            if mode=='test':
                df = df.drop([
                    '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Basal ganglia', 'Brain stem', 'Cerebellum', 'Brain stem','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS'], axis=1)
            else:
                df = df.drop([
                    '(ARE)', 'pre CO  H', 'EVD', 'emb', 'cran', 'hem',
                    'Brain stem', 'Cerebellum', 'Brain stem',
                    'Basal ganglia','Corpus callosum', 'Hemisphere', 'Lat ventricle', 'thalamus',
        'GKS','index'], axis=1)
                
    df['Age at GK'] = scaler.fit_transform(df[['Age at GK']])
    df['CSF體積(ml)'] = scaler.fit_transform(df[['CSF體積(ml)']])
    df['腦組織體積(ml)'] = scaler.fit_transform(df[['腦組織體積(ml)']])
    df['血管體積(ml)'] = scaler.fit_transform(df[['血管體積(ml)']])
    df['TC(Gy)'] = scaler.fit_transform(df[['TC(Gy)']])
    df['TP(Gy)'] = scaler.fit_transform(df[['TP(Gy)']])
    df['RV'] = scaler.fit_transform(df[['RV']])
                
    return df 
    
if __name__=='__main__':
      
      df=pd.read_excel('./Dataset/TabularData_pattern/fold0_train.xlsx')
      df=pre_tabular(df,'pattern ARE','train')
      print(df)