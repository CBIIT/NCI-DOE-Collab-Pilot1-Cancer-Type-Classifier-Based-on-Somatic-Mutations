### Model description:
P1B2 is an a deep learning network that can classify the cancer type using patient somatic SNPs.
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
   python p1b2_baseline_keras2.py --val_split 0.2
   ```
The training and test data files will be downloaded the first time this is run and will be cached for future runs.

#### Example output

```
Using TensorFlow backend.
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
Train on 2400 samples, validate on 600 samples
Epoch 1/20
2400/2400 [==============================] - 6s 2ms/step - loss: 2.2638 - acc: 0.1979 - val_loss: 2.0995 - val_acc: 0.1850
Epoch 2/20
2400/2400 [==============================] - 2s 672us/step - loss: 1.7935 - acc: 0.3933 - val_loss: 1.8121 - val_acc: 0.4150
Epoch 3/20
2400/2400 [==============================] - 2s 670us/step - loss: 1.2672 - acc: 0.6325 - val_loss: 1.8914 - val_acc: 0.3433
Epoch 4/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.8658 - acc: 0.7892 - val_loss: 1.5896 - val_acc: 0.5267
Epoch 5/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.6078 - acc: 0.8912 - val_loss: 1.5906 - val_acc: 0.5483
Epoch 6/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.4413 - acc: 0.9463 - val_loss: 1.9003 - val_acc: 0.4950
Epoch 7/20
2400/2400 [==============================] - 2s 670us/step - loss: 0.3382 - acc: 0.9746 - val_loss: 2.0737 - val_acc: 0.4917
Epoch 8/20
2400/2400 [==============================] - 2s 674us/step - loss: 0.2894 - acc: 0.9817 - val_loss: 2.3095 - val_acc: 0.4950
Epoch 9/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.2524 - acc: 0.9871 - val_loss: 1.6557 - val_acc: 0.5550
Epoch 10/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.2306 - acc: 0.9888 - val_loss: 2.1757 - val_acc: 0.5133
Epoch 11/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.2159 - acc: 0.9913 - val_loss: 1.9694 - val_acc: 0.5350
Epoch 12/20
2400/2400 [==============================] - 2s 675us/step - loss: 0.2064 - acc: 0.9892 - val_loss: 1.7639 - val_acc: 0.5250
Epoch 13/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.2090 - acc: 0.9867 - val_loss: 1.6637 - val_acc: 0.5667
Epoch 14/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.1848 - acc: 0.9938 - val_loss: 1.8245 - val_acc: 0.5067
Epoch 15/20
2400/2400 [==============================] - 2s 674us/step - loss: 0.1747 - acc: 0.9942 - val_loss: 1.7214 - val_acc: 0.5533
Epoch 16/20
2400/2400 [==============================] - 2s 675us/step - loss: 0.1788 - acc: 0.9917 - val_loss: 1.6534 - val_acc: 0.5833
Epoch 17/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.1758 - acc: 0.9913 - val_loss: 2.2487 - val_acc: 0.5050
Epoch 18/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.1674 - acc: 0.9929 - val_loss: 2.1581 - val_acc: 0.5217
Epoch 19/20
2400/2400 [==============================] - 2s 672us/step - loss: 0.1627 - acc: 0.9942 - val_loss: 1.7176 - val_acc: 0.5767
Epoch 20/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.1791 - acc: 0.9829 - val_loss: 1.6960 - val_acc: 0.5783
Saved json model to disk
best_val_loss = 1.58964 best_val_acc = 0.58333
Evaluation on test data: {'accuracy': 0.543}
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
Evaluation on test data: {'accuracy': 0.544}

```
