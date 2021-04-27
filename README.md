# NCI-DOE-Collab-Pilot1-Cancer-Type-Classifier-Based-on-Somatic-Mutations

### Description
The Cancer Type Classifier Based on Somatic Mutations capability (Pilot 1 Benchmark 2, also known as P1B2) shows how to build a deep learning network that can classify the cancer type from somatic mutations.

### User Community
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability
A data scientist can use the provided untrained model as reference, or the trained model for inference. The provided scripts use data that has been downloaded and processed from the Genomic Data Commons (GDC).

### Uniqueness
This model aggregates the variation impact by gene from 2.7 million unique single nucleotide polymorphisms (SNPs), which might is one of many methods to reduce the features space. 

### Components
The following components are in the [Cancer Type Classifier Based on Somatic Mutations (P1B2)
](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7564992) dataset in the Model and Data Clearinghouse (MoDaC):
* Untrained model: 
  * The untrained neural network model is defined in the model topology file p1b2.model.json. 
* Data:
  * The processed training and test data are in MoDaC.
* Trained model:
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference p1b2.model.h5.

### Technical Details
Refer to this [README](./Pilot1/P1B2/README.md).
