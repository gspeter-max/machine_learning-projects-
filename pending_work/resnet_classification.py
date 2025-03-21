import numpy as np

class resnet_classification:

    def __init__(self, kernel: (np.ndarray, list) = None):
        if kernel is None:
            self.kernel = np.array([[
                [-1,0,1],
                [-1,0,1],
                [-1,0,1]
            ],
               [[-1,0,1],
                [-1,0,1],
                [-1,0,1]],

            [[-1,0,1],
                [-1,0,1],
                [-1,0,1]]]
    )

    def Batchnormalization(self, input):

        '''
        1st compute the mean and  std ( for each filter  becuase batchnormalization means each filter or layernormalization  means each sample )
        2st use normalization formula ( z_score, min-max)
        and  do that
        '''
        batch_mean = np.mean(input , axis = (0,1,2), keepdims = True)
        batch_std = np.std(input, axis = (0,1,2) , keepdims = True)
        upper =input - batch_mean
        lower = np.sqrt((batch_std + 1e-4)**2)
        return  upper / lower

    def relu(self,input):
        x =  np.maximum(0, input)
        return x.shape, x

    def conv2D(self,input,input_shape ,old_filter = None,filter= 10 ,kernel_size = 3, kernel= None, stride = 1, padding = 'valid'):
        if old_filter is None:
            old_filter = filter

        if kernel is None:
            kernel = np.random.randn(kernel_size,kernel_size,old_filter)

        final_features = []
        if padding == "same":

            if stride != 1 :
                raise ValueError('make sure stride 1 for padding = "same"')

            padding = ((kernel_size - 1 )// 2 )
            output_size = (input_shape[1] + kernel_size - 2 * padding - 1 ) // stride


        else:

            output_size = int(((input_shape[1] - kernel_size + 1 )) // stride)


        for _ in range(filter):

            each_filter_featuers = np.zeros((input_shape[0],output_size , output_size  ))

            for i in range(output_size-2):

                for j in range(output_size-2):
                    values = input[:,j * stride : j * stride + kernel_size, i * stride : i* stride + kernel_size, :] * kernel
                    each_filter_featuers[:,j,i] = np.sum(values, axis = (1,2,3))

            final_features.append(each_filter_featuers)
        final_features = np.array(final_features)
        final_features = np.transpose(final_features, axes = (1,2,3,0))

        return final_features

    # i that is restnet-30
    def residual_block(self,input, filter, stride, padding,input_shape,first):
    
        if first : 
            x = self.conv2D(input = input,input_shape = input_shape, filter=filter, stride=stride, padding=padding,kernel=self.kernel)
        
        else: 
            x = self.conv2D(input = input,input_shape = input_shape, filter=filter, stride=stride, padding=padding)

        x = self.Batchnormalization(x)
        shape, x1= self.relu(x)

        x = self.conv2D(input = x1,input_shape = shape, filter= filter,stride= stride, padding= 'same')
        x = self.Batchnormalization(x)
        x = x + x1

        output_shape,x = self.relu(x) 
        return output_shape,x

    def global_avg_pooling(self,x): 
        x = np.mean(x, axis= (1,2), keepdims = False)
        return x.shape,x 
    
    def dense(self, input_shape, output_shape, x): 
        w = np.random.randn(input_shape[1],output_shape)
        output = np.dot(x,w)
        return output

    def softmax(self,x): 
        values = np.exp(x - np.max(x, axis = 1, keepdims = True)) 
        output  = values / np.sum(values, axis = 0 )        
        return output
    
    
    def final_layer(self,input,filter,stride, padding ,input_shape,output_node = 10):

        output_s,x = self.residual_block(input,filter,stride,padding,input_shape,first = True)
        output_s,x = self.residual_block(x,filter,stride,padding,output_s,first = False)
        x2_shape,x = self.global_avg_pooling(x) 
        x = self.dense(x2_shape,output_node,x)
        x = self.softmax(x)
        return x

y = restnet_classificaiton()
x = y.final_layer(x_train,10,stride=1, padding='valid',input_shape=(5, 32, 32, 3))
