# NCI-DOE-Collab-Pilot1-Mutation_Classifier

### Description:
The Mutation Classifier capability (P1B2) shows how to build a deep learning network that can classify the cancer type.

### User Community:	
Primary: Cancer biology data modeling</br>
Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The provided untrained model can be used by a data scientist. The scripts use already processed data.

### Uniqueness:	
Using neural networks in classification of somatic mutation has been presented in other research papers. This model aggregates the variation impact by gene from 2.7 million unique SNPs which might not be the best way to reduce the features space. The technical team is not sure about the uniqueness of the method used to reduce the dimension of the somatic mutations.

### Components:	

Untrained model: 
* Untrained neural network model is defined in [p1b2.model.json](https://modac.cancer.gov/searchTab?dme_data_id=).

Data:
* Processed training and test data in [ToDo: MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=).

Trained Model:
* Trained model is defined by combining the untrained model + model weights.
* Trained model weights are used in inference [p1b2.model.h5](https://modac.cancer.gov/searchTab?dme_data_id=).

### Technical Details:
Please refer to this [README](./Pilot1/P1B2/README.md).
