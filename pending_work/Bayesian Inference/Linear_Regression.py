from auto_process_data import datas 

x,y = datas.load_data() 
''' 
1, comppute the inital prior information ( parameters) ( but that get overfit , normal distribution that take )
2. likehood = ( features , house_prices , location)
3. p(d / p) * p( prior ) ==>  that to update you belives 


'''
prrior = np.random.normal()