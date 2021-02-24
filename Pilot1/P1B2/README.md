### Model description:
P1B2, an a deep learning network that can classify the cancer type using patient somatic SNPs.
The proposed network architecture is MLP with regularization, which includes 5 layers.
The model is trained and validated on SNP data from GDC. The sample size is 4,000 (3000 training + 1000 test).
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
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
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
Train on 2400 samples, validate on 600 samples
Epoch 1/20
2400/2400 [==============================] - 11s 5ms/step - loss: 2.3507 - acc: 0.1762 - val_loss: 2.1590 - val_acc: 0.1983
Epoch 2/20
2400/2400 [==============================] - 2s 733us/step - loss: 1.9286 - acc: 0.3563 - val_loss: 1.9081 - val_acc: 0.3200
Epoch 3/20
2400/2400 [==============================] - 2s 731us/step - loss: 1.2972 - acc: 0.6208 - val_loss: 1.6839 - val_acc: 0.4317
Epoch 4/20
2400/2400 [==============================] - 2s 733us/step - loss: 0.7665 - acc: 0.8417 - val_loss: 1.5829 - val_acc: 0.5417
Epoch 5/20
2400/2400 [==============================] - 2s 733us/step - loss: 0.4754 - acc: 0.9558 - val_loss: 1.5494 - val_acc: 0.5433
Epoch 6/20
2400/2400 [==============================] - 2s 733us/step - loss: 0.3217 - acc: 0.9925 - val_loss: 1.5524 - val_acc: 0.5417
Epoch 7/20
2400/2400 [==============================] - 2s 732us/step - loss: 0.2715 - acc: 0.9958 - val_loss: 1.5190 - val_acc: 0.5533
Epoch 8/20
2400/2400 [==============================] - 2s 732us/step - loss: 0.2442 - acc: 0.9967 - val_loss: 1.4909 - val_acc: 0.5667
Epoch 9/20
2400/2400 [==============================] - 2s 730us/step - loss: 0.2242 - acc: 0.9979 - val_loss: 1.4973 - val_acc: 0.5500
Epoch 10/20
2400/2400 [==============================] - 2s 728us/step - loss: 0.2094 - acc: 0.9975 - val_loss: 1.5475 - val_acc: 0.5283
Epoch 11/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1964 - acc: 0.9988 - val_loss: 1.4513 - val_acc: 0.5683
Epoch 12/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1867 - acc: 0.9988 - val_loss: 1.5095 - val_acc: 0.5483
Epoch 13/20
2400/2400 [==============================] - 2s 730us/step - loss: 0.1775 - acc: 0.9988 - val_loss: 1.5169 - val_acc: 0.5383
Epoch 14/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1710 - acc: 0.9988 - val_loss: 1.4858 - val_acc: 0.5500
Epoch 15/20
2400/2400 [==============================] - 2s 737us/step - loss: 0.1637 - acc: 0.9996 - val_loss: 1.4768 - val_acc: 0.5600
Epoch 16/20
2400/2400 [==============================] - 2s 734us/step - loss: 0.1585 - acc: 0.9996 - val_loss: 1.4480 - val_acc: 0.5633
Epoch 17/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1537 - acc: 0.9996 - val_loss: 1.4665 - val_acc: 0.5650
Epoch 18/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1493 - acc: 0.9996 - val_loss: 1.4576 - val_acc: 0.5617
Epoch 19/20
2400/2400 [==============================] - 2s 729us/step - loss: 0.1443 - acc: 0.9996 - val_loss: 1.4491 - val_acc: 0.5650
Epoch 20/20
2400/2400 [==============================] - 2s 731us/step - loss: 0.1417 - acc: 0.9996 - val_loss: 1.4648 - val_acc: 0.5650
Saved json model to disk
best_val_loss=1.44796 best_val_acc=0.56833
Best model saved to: model.A=sigmoid.B=64.D=None.E=20.L1=1024.L2=512.L3=256.P=1e-05.h5
Evaluation on test data: {'accuracy': 0.56}
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
Evaluation on test data: {'accuracy': 0.55}
```
