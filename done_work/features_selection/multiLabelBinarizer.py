import numpy as np

def multiLabelBinarizer(index_col, columns_col , data : pd.DataFrame): 

    if not isinstance(data,pd.DataFrame): 
        raise ValueError(' data must be {   pd.DataFrame  }')
    
    data = data.drop_duplicates(index_col)
    columns = list({  interest  for sublist in data[columns_col] for interest in sublist})

    # THAT IS NOT MUCH EFFICIENT 
    # for sub_interest in data[columns_col]: 
    #     for i in range(len(sub_interest)): 
    #         if not sub_interest[i] in columns: 
    #             columns.append(sub_interest[i]) 


    df = pd.DataFrame(0 ,columns= columns, index= data[index_col], dtype=np.int8)

    for i , col_name in enumerate(columns): 
        for index , interest in enumerate(data[columns_col]): 

            if col_name in interest: 
                df.iloc[index,i] = 1 

    return df  

df = multiLabelBinarizer('user_id','interests',users_df)
print(df)
