from sklearn.preprocessing import LabelEncoder , LabelBinarizer , OneHotEncoder



def EncodeLabel(data , feature , binary=True , OneHot=False):
    """
    a function to encode the categorical fearures of the data
        data(pandas.DataFrame) : target dataframe
        feature(str) : feature to be encoded
        binary(bool) : True if the feature is binary  else False
        OneHot(bool) : set it True to encode the non binary varibales in onehot encoding 
        format.

    returns
        sklearn.preprocessing.OneHotEncoder : if OneHot is True , This encoder is fitted to 
        the data and can be used to convert it back to its original form if needed using the
        reverse_transform function.
    """
    if binary:
        lb = LabelBinarizer()
        temp = lb.fit_transform(data[feature])
        data[feature] = temp
        print(f'data type of {feature} column = {data[feature].dtypes}')
        print(f'unique values in {feature} = {data[feature].unique()}')
    else:
        if OneHot:
            # creating the starting string of Onehot column names
            start= None
            if start is None:
                words = feature.split(' ')
                if len(words) > 1:
                    for word in words:
                        start += word[0]
            if start is None:
                words = feature.split('_')
                if len(words) > 1:
                    for word in words:
                        start += word[0]
            if start is None:
                for a in feature:
                    if a.ispper():
                        start+=a
            if start is None:
                start = feature[0:4]
            
            # creating column names 
            ds = {a:i for i, a in data[feature].unique()}
            cols = [start+'_'+str(ds[a]) for a in feature]

            # one hot encode the data
            oh = OneHotEncoder()
            temp = pd.DataFrame(oh.fit_transform(data[[feature]]).toarray() , columns=cols)
            data = data.join(temp)
            data.drop(columns=[feature])
            print(f'number of columns added  = {data[feature].nunique()}')
            print(f'column dropped = {feature}')
            print(f'name of new columns = {cols}')
            return oh
        else:
            le = LabelEncoder()
            temp = le.fit_transform(data[feature])
            data[feature] = temp
            print(f'data type of {feature} column = {data[feature].dtypes}')
            print(f'unique values in {feature} = {data[feature].unique()}')