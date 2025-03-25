encoder = target_encoder(strategy = 'auto',data = df_pandas ,
                         categorical_featues = ['cat_feature_0', 'cat_feature_1',
       'cat_feature_2', 'cat_feature_3', 'cat_feature_4', 'cat_feature_5',
       'cat_feature_6', 'cat_feature_7', 'cat_feature_8', 'cat_feature_9'] , 
       target_feature = 'target' ) 