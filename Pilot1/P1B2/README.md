## P1B2: Sparse Classifier Disease Type Prediction from Somatic SNPs

### Model description:
P1B2, an a deep learning network that can classify the cancer type using patient somatic SNPs.
The proposed network architecture is MLP with regularization, which includes 5 layers.
The model is trained and cross-validated on SNP data from GDC. The sample size is 4,000 (3000 training + 1000 test).
The full set of features contains 28,205 columns.
It is useful for classification based on very sparse input data and evaluation of the information content and predictive value in a molecular assay with auxiliary learning tasks.

### Description of the Data:
* Data source: SNP data from GDC MAF files
* Input dimensions: 28,205 (aggregated variation impact by gene from 2.7 million unique SNPs)
* Output dimensions: 10 class probabilities (9 most abundant cancer types in GDC + 1 “others”)
* Sample size: 4,000 (3000 training + 1000 test)
* Notes on data balance and other issues: data balance achieved via undersampling; “others” category drawn from all remaining lower-abundance cancer types in GDC

### Expected Outcomes:
* Classification
* Output range or number of classes: 10

### Setup:
To setup the python environment needed to train and run this model, first make sure you install [conda](https://docs.conda.io/en/latest/) package manager, clone this repository, then create the environment as shown below.

```bash
   conda env create -f environment.yml -n P1B2
   conda activate P1B2
   ```
   
To download the processed data needed to train and test the model, and the trained model files, you should create an account first on the Model and Data Clearinghouse [MoDac](modac.cancer.gov). The training and test scripts will prompt you to enter your MoDac credentials.

### Training:

To train the model from scratch, the script [p1b2_baseline_keras2.py](p1b2_baseline_keras2.py) does the following:
* Reads the model configuration parameters from [p1b2_default_model.txt](p1b2_default_model.txt)
* Downloads the training data and splits it to training/validation sets
* Creates and trains the keras model
* Saves the best trained model based on the model performance on the validation dataset
* Evaluates the best model on the test dataset

```cd Pilot1/P1B2
   python p1b2_baseline_keras2.py
   ```
The training and test data files will be downloaded the first time this is run and will be cached for future runs.

#### Example output

```
Using TensorFlow backend.

Shape X_train:  (2700, 28204)
Shape X_val:  (300, 28204)
Shape X_test:  (1000, 28204)
Shape y_train:  (2700, 10)
Shape y_val:  (300, 10)
Shape y_test:  (1000, 10)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 28204)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              28881920  
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_4 (Dense)              (None, 10)                2570      
=================================================================
Total params: 29,540,618
Trainable params: 29,540,618
Non-trainable params: 0
____________________________________________________________________________________________________
None
Train on 2700 samples, validate on 300 samples
Epoch 1/20
2700/2700 [==============================] - 11s 4ms/step - loss: 2.3163 - acc: 0.1637 - val_loss: 6.0741 - val_acc: 0.0000e+00
Epoch 2/20
2700/2700 [==============================] - 3s 1ms/step - loss: 2.0578 - acc: 0.2607 - val_loss: 7.7469 - val_acc: 0.0000e+00
Epoch 3/20
2700/2700 [==============================] - 3s 1ms/step - loss: 1.8499 - acc: 0.3478 - val_loss: 9.5266 - val_acc: 0.0000e+00
Epoch 4/20
2700/2700 [==============================] - 3s 1ms/step - loss: 1.6315 - acc: 0.4374 - val_loss: 11.1787 - val_acc: 0.0000e+00
Epoch 5/20
2700/2700 [==============================] - 3s 1ms/step - loss: 1.4228 - acc: 0.5044 - val_loss: 13.1909 - val_acc: 0.0000e+00
Epoch 6/20
2700/2700 [==============================] - 3s 1ms/step - loss: 1.2675 - acc: 0.5833 - val_loss: 15.7211 - val_acc: 0.0000e+00
Epoch 7/20
2700/2700 [==============================] - 3s 1ms/step - loss: 1.1087 - acc: 0.6819 - val_loss: 16.3223 - val_acc: 0.0000e+00
Epoch 8/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.9924 - acc: 0.7159 - val_loss: 16.3702 - val_acc: 0.0000e+00
Epoch 9/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.9099 - acc: 0.7452 - val_loss: 16.3683 - val_acc: 0.0000e+00
Epoch 10/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.8459 - acc: 0.7759 - val_loss: 16.3711 - val_acc: 0.0000e+00
Epoch 11/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.7872 - acc: 0.8022 - val_loss: 16.3704 - val_acc: 0.0000e+00
Epoch 12/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.7390 - acc: 0.8189 - val_loss: 16.3710 - val_acc: 0.0000e+00
Epoch 13/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.7014 - acc: 0.8456 - val_loss: 16.3687 - val_acc: 0.0000e+00
Epoch 14/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.6254 - acc: 0.8830 - val_loss: 16.3755 - val_acc: 0.0000e+00
Epoch 15/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.5848 - acc: 0.9041 - val_loss: 16.3745 - val_acc: 0.0000e+00
Epoch 16/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.5538 - acc: 0.9081 - val_loss: 16.3750 - val_acc: 0.0000e+00
Epoch 17/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.4993 - acc: 0.9330 - val_loss: 16.3755 - val_acc: 0.0000e+00
Epoch 18/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.5186 - acc: 0.9174 - val_loss: 16.3738 - val_acc: 0.0000e+00
Epoch 19/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.4608 - acc: 0.9396 - val_loss: 16.3709 - val_acc: 0.0000e+00
Epoch 20/20
2700/2700 [==============================] - 3s 1ms/step - loss: 0.4824 - acc: 0.9330 - val_loss: 16.3686 - val_acc: 0.0000e+00

best_val_loss = 6.07406 best_val_acc = 0.00000

Evaluation on test data: {'accuracy': 0.475}
```

### Preliminary performance:

The XGBoost classifier below achieves ~55% average accuracy on validation data in the five-fold cross validation experiment. This suggests there may be a low ceiling for the MLP results; there may not be enough information in this set of SNP data to classify cancer types accurately.

```cd Pilot1/P1B2
   python p1b2_xgboost.py
   ```

### Inference: 

To test the trained model in inference, the script [p1b2_infer.py](p1b2_infer.py) does the following:
* Loads the trained model
* Downloads the processed test dataset with the corresponding labels
* Performs inference on the test dataset
* Reports the accuracy of the model on the test dataset

```bash
   python p1b2_infer.py
   ```
#### Example output
```
Evaluation on test data: {'accuracy': 0.475}
```
