import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler # Added import for the example usage

class DeepQLearning:
    def __init__(self, input_shape, lasso_lambda ,ridge_lambda , num_actions):
        self.input_shape = input_shape
        self.ridge_lambda = ridge_lambda
        self.lasso_lambda = lasso_lambda
        self.num_actions = num_actions  # Number of possible actions
        self.model = self.build_model()
        self.target_model = tf.keras.models.clone_model(self.model)  
        # Ensure the model is built before cloning if using subclassed model (not strictly needed for Functional)
        # self.model.build((None,) + input_shape if isinstance(input_shape, tuple) else (None, input_shape)) 
        # self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def mse(self, y_true, y_pred):
        # Ensure inputs are float32 for calculations
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def compute_loss(self, y_true, y_pred):
        # Ensure lambdas are float32
        lasso_lambda_f32 = tf.cast(self.lasso_lambda, dtype=tf.float32)
        ridge_lambda_f32 = tf.cast(self.ridge_lambda, dtype=tf.float32)
        
        mse_loss = self.mse(y_true, y_pred)
        
        # Calculate L1 regularization loss
        lasso_loss = tf.constant(0.0, dtype=tf.float32)
        if lasso_lambda_f32 > 0:
             l1_term = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_weights if 'kernel' in w.name]) # Optional: only kernels
             # l1_term = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_weights]) # Original: all weights
             lasso_loss = tf.multiply(lasso_lambda_f32, l1_term)

        # Calculate L2 regularization loss
        ridge_loss = tf.constant(0.0, dtype=tf.float32)
        if ridge_lambda_f32 > 0:
            l2_term = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights if 'kernel' in w.name]) # Optional: only kernels
            # l2_term = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights]) # Original: all weights
            ridge_loss = tf.multiply(ridge_lambda_f32, l2_term)
            
        return mse_loss + lasso_loss + ridge_loss

    def build_model(self):
        # Ensure input_shape is correctly handled (e.g., if it's just an int)
        shape_tuple = (self.input_shape,) if isinstance(self.input_shape, int) else self.input_shape 
        
        inputs = tf.keras.layers.Input(shape=shape_tuple)
        x1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
        x2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Dropout(0.2)(x2) # Apply dropout after normalization usually

        # Ensure skip connection matches dimensions if needed, here it seems okay
        skip = tf.keras.layers.Dense(128, activation='linear', kernel_initializer='he_normal')(x2) 
        x2_dense = tf.keras.layers.Dense(128, activation='linear', kernel_initializer='he_normal')(x2) # Renamed for clarity

        x3 = tf.keras.layers.Add()([x2_dense, skip])
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x3 = tf.keras.layers.ReLU()(x3) # Apply activation after batch norm and add

        output = tf.keras.layers.Dense(self.num_actions, activation='linear', kernel_initializer='he_normal')(x3)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        # No compilation needed here as we use a custom loop
        return model
    
    def update_model_weight(self, tau):
        # Ensure tau is float32
        tau = tf.cast(tau, dtype=tf.float32)
        for target_weight, model_weight in zip( self.target_model.trainable_weights, self.model.trainable_weights ):
            target_weight.assign( tau * model_weight + ( 1.0 - tau) * target_weight) # Use 1.0 for float


    def model_train(self, x_train, y_train, epochs=100, batch_size=200):
        # Ensure data is float32
        x_train = tf.cast(x_train, dtype=tf.float32)
        y_train = tf.cast(y_train, dtype=tf.float32)

        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_mse_avg = tf.keras.metrics.Mean()

            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    # Ensure model is called with training=True for Dropout/BatchNorm
                    y_pred = self.model(x_batch, training=True) 
                    # Calculate loss using the custom function
                    loss = self.compute_loss(y_batch, y_pred)
                    # Calculate plain MSE for reporting
                    mse = self.mse(y_batch, y_pred) 
                
                # Get gradients and update weights
                gradients = tape.gradient(loss, self.model.trainable_variables)
                # Optional: Gradient Clipping
                # gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Update metrics
                epoch_loss_avg.update_state(loss)
                epoch_mse_avg.update_state(mse)

                # Update target network (NOTE: Frequency and Tau value are critical for actual RL)
                # Consider updating less frequently or with smaller tau for RL
                self.update_model_weight(tau = 0.005) # Using a more standard small tau value for soft updates


            print(f'Epoch {epoch}: Loss = {epoch_loss_avg.result().numpy()} --- Mse = {epoch_mse_avg.result().numpy()}')

# --- Example Testing setup ---
input_shape = 12
num_actions = 4  # Define the number of actions/outputs
batch_size = 200 # Corrected batch size for training call

# Generate random training data (as in your example)
X_train_np = np.random.randn(1000, input_shape).astype(np.float32)
# Target shape should match num_actions
y_train_np = np.random.randn(1000, num_actions).astype(np.float32)  

# --- Preprocessing as in your example ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train_np)

# Normalize targets (NOTE: Usually not done for Q-values in RL)
y_mean = np.mean(y_train_np, axis=0) # Calculate mean per action/output
y_std = np.std(y_train_np, axis=0)   # Calculate std dev per action/output
# Add epsilon to prevent division by zero if std dev is 0
y_train_normalized = (y_train_np - y_mean) / (y_std + 1e-8) 
# --- End Preprocessing ---


# Create and train the model instance
temp_model = DeepQLearning(
    input_shape=input_shape, 
    lasso_lambda=0.0001, # Example L1 strength
    ridge_lambda=0.0001, # Example L2 strength
    num_actions=num_actions
)

# Call the training method
temp_model.model_train(x_train_scaled, y_train_normalized, epochs=100, batch_size=batch_size)
