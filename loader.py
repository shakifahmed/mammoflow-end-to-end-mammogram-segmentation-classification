import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class Loader:

    def data_loader(aug_dir):
        images = []
        categories = os.listdir(aug_dir)
        categories.sort()
        print('Subfolders:')
        for i in range(len(categories)):
            print(categories[i])

        for category in categories:
            path = os.path.join(aug_dir,category)
            num_category = categories.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img),0)
                images.append([img_array,num_category])
        return images

    def separator(images):
        X = []
        y = []
        for features, labels in images:
            X.append(features)
            y.append(labels)
        X = np.array(X,np.float32).reshape(-1, 224, 224, 1)
        y = np.array(y,np.float32)
        return X, y
    
    def dataframe(X, y):
        X = X.reshape(X.shape[0],-1)
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)

        df_X['feature_id'] = range(0, len(df_X))
        df_X.insert(loc=0, column='feature_id', value=df_X.pop('feature_id'))

        df_y['label_id'] = range(0,len(y))
        df_y.insert(loc=0, column='label_id', value=df_y.pop('label_id'))

        df = df_y.merge(df_X,left_on='label_id',right_on='feature_id',how='inner')

        #drop extra column
        df.drop(['label_id', 'feature_id'], axis=1,inplace=True) 

        #renamne column 0_x as 'label' and 0_y as '0'
        df = df.rename(columns={'0_x':'label','0_y':'0'})
        # df.to_csv("/teamspace/studios/this_studio/mydata.csv", index=False)

        return df

    def split(df):
        X_train,X_test,y_train,y_test = train_test_split(
            df.iloc[:,1:df.shape[1]],
            df.iloc[:,0],
            test_size=0.2,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def data_balance(X_train, X_test, y_train):
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
        smote = SMOTE(random_state=42,sampling_strategy = 'not majority')
        X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train
    
    def normalize(X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns) #columns will be X_train cause X_train_scaled has no col
        X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

        return X_train_scaled,  X_test_scaled
    
    def revert(X_train_scaled,  X_test_scaled):
        X_train_scaled_dl = X_train_scaled.values.reshape(-1, 224, 224, 1)
        X_test_scaled_dl = X_test_scaled.values.reshape(-1, 224, 224, 1)

        return X_train_scaled_dl, X_test_scaled_dl
