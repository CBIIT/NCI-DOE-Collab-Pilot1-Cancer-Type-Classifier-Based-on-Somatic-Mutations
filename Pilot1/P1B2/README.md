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
2400/2400 [==============================] - 5s 2ms/step - loss: 2.3920 - acc: 0.1517 - val_loss: 2.1867 - val_acc: 0.1750
Epoch 2/20
2400/2400 [==============================] - 2s 764us/step - loss: 2.0228 - acc: 0.2825 - val_loss: 1.9870 - val_acc: 0.2733
Epoch 3/20
2400/2400 [==============================] - 2s 764us/step - loss: 1.4299 - acc: 0.5875 - val_loss: 1.7429 - val_acc: 0.4633
Epoch 4/20
2400/2400 [==============================] - 2s 767us/step - loss: 0.8213 - acc: 0.8117 - val_loss: 1.5820 - val_acc: 0.5433
Epoch 5/20
2400/2400 [==============================] - 2s 764us/step - loss: 0.5531 - acc: 0.8975 - val_loss: 1.5210 - val_acc: 0.5367
Epoch 6/20
2400/2400 [==============================] - 2s 763us/step - loss: 0.3607 - acc: 0.9808 - val_loss: 1.5366 - val_acc: 0.5683
Epoch 7/20
2400/2400 [==============================] - 2s 765us/step - loss: 0.2781 - acc: 0.9954 - val_loss: 1.5275 - val_acc: 0.5717
Epoch 8/20
2400/2400 [==============================] - 2s 763us/step - loss: 0.2452 - acc: 0.9971 - val_loss: 1.4767 - val_acc: 0.5733
Epoch 9/20
2400/2400 [==============================] - 2s 766us/step - loss: 0.2249 - acc: 0.9979 - val_loss: 1.4953 - val_acc: 0.5667
Epoch 10/20
2400/2400 [==============================] - 2s 764us/step - loss: 0.2082 - acc: 0.9988 - val_loss: 1.4795 - val_acc: 0.5583
Epoch 11/20
2400/2400 [==============================] - 2s 763us/step - loss: 0.1962 - acc: 0.9988 - val_loss: 1.4968 - val_acc: 0.5633
Epoch 12/20
2400/2400 [==============================] - 2s 765us/step - loss: 0.1859 - acc: 0.9988 - val_loss: 1.5350 - val_acc: 0.5483
Epoch 13/20
2400/2400 [==============================] - 2s 765us/step - loss: 0.1775 - acc: 0.9992 - val_loss: 1.4610 - val_acc: 0.5700
Epoch 14/20
2400/2400 [==============================] - 2s 762us/step - loss: 0.1708 - acc: 0.9992 - val_loss: 1.4659 - val_acc: 0.5817
Epoch 15/20
2400/2400 [==============================] - 2s 761us/step - loss: 0.1639 - acc: 0.9996 - val_loss: 1.4894 - val_acc: 0.5717
Epoch 16/20
2400/2400 [==============================] - 2s 787us/step - loss: 0.1580 - acc: 0.9996 - val_loss: 1.5002 - val_acc: 0.5700
Epoch 17/20
2400/2400 [==============================] - 2s 769us/step - loss: 0.1535 - acc: 0.9996 - val_loss: 1.4900 - val_acc: 0.5750
Epoch 18/20
2400/2400 [==============================] - 2s 764us/step - loss: 0.1493 - acc: 0.9996 - val_loss: 1.4812 - val_acc: 0.5750
Epoch 19/20
2400/2400 [==============================] - 2s 761us/step - loss: 0.1452 - acc: 0.9996 - val_loss: 1.4800 - val_acc: 0.5700
Epoch 20/20
2400/2400 [==============================] - 2s 765us/step - loss: 0.1416 - acc: 0.9996 - val_loss: 1.4625 - val_acc: 0.5767
Saved json model to disk
best_val_loss=1.46097 best_val_acc=0.58167
Best model saved to: model.A=sigmoid.B=64.D=None.E=20.L1=1024.L2=512.L3=256.P=1e-05.h5
Evaluation on test data: {'accuracy': 0.557}
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
