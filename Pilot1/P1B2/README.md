### Model Description
The Cancer Type Classifier Based on Somatic Mutations capability (Pilot 1 Benchmark 2, also known as P1B2) is a deep learning network that can classify the cancer type using patient somatic Single Nucleotide Polymorphisms (SNPs). The proposed network architecture is MultiLayer Perceptron (MLP) with regularization, which includes five layers. We trained and validated the model on SNP data from Genomic Data Commons (GDC). It is useful for classification based on very sparse input data and evaluation of the information content and predictive value in a molecular assay with auxiliary learning tasks.

### Description of the Data
* Data source: SNP data from GDC Mutation Annotation Format (MAF) files
* Input dimensions: 28,205 columns (aggregated variation impact by gene from 2.7 million unique SNPs) 
* Output dimensions: 10 class probabilities (the nine most abundant cancer types in GDC and one probability for “others”)
* Sample size: 4,000 samples (3000 training and 1000 test) 
* Notes on data balance and other issues: Data balance achieved via undersampling; The “others” category represents all remaining lower-abundance cancer types in GDC.

### Expected Outcomes
* Classification
* Output range or number of classes: 10

### Setup
To set up the Python environment needed to train and run this model:
1. Install the [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n P1B2
   conda activate P1B2
   ```
   
To download the processed data needed to train and test the model, and the trained model files:
1. Create an account on the Model and Data Clearinghouse ([MoDaC](https://modac.cancer.gov/)). 
2. Follow the instructions in the Training section below.
3. When prompted by the training and test scripts, enter your MoDaC credentials.

### Training

To train the model from scratch, execute the script [p1b2_baseline_keras2.py](p1b2_baseline_keras2.py), as follows: 

```cd Pilot1/P1B2
   python p1b2_baseline_keras2.py --val_split 0.2 --epochs 20
   ```

This script does the following:
* Reads the model configuration parameters from [p1b2_default_model.txt](p1b2_default_model.txt)
* Downloads the training data and splits it into training/validation sets.
* Creates and trains the Keras model.
* Saves the best trained model based on the model performance on the validation dataset.
* Evaluates the best model on the test dataset.

The first time you run the script, it downloads the training and test data files. Then it caches the files for future runs.

#### Example Output

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
......
......
Epoch 20/20
2400/2400 [==============================] - 2s 671us/step - loss: 0.1791 - acc: 0.9829 - val_loss: 1.6960 - val_acc: 0.5783
Saved json model to disk
best_val_loss = 1.58964 best_val_acc = 0.58333
Evaluation on test data: {'accuracy': 0.543}
```

### Inference

To test the trained model in inference, execute the script [p1b2_infer.py](p1b2_infer.py), as follows: 

```bash
   python p1b2_infer.py
   ```

This script does the following:
* Downloads the trained model from MoDaC. 
* Downloads the processed test dataset from MoDaC with the corresponding labels.
* Performs inference on the test dataset.
* Reports the accuracy of the model on the test dataset.

#### Example Output
```
Evaluation on test data: {'accuracy': 0.544}

```
