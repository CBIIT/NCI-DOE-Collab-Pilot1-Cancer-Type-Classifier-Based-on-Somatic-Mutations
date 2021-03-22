# NCI-DOE-Collab-Pilot1-Mutation_Classifier

### Description:
The Mutation Classifier capability (Pilot 1 Benchmark 2, also known as P1B2) shows how to build a deep learning network that can classify the cancer type.

### User Community:	
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology


### Usability:	
The provided untrained model can be used by a data scientist. The scripts use already processed data.

### Uniqueness:	
Using neural networks in classification of somatic mutation has been presented in other research papers. This model aggregates the variation impact by gene from 2.7 million unique single nucleotide polymorphisms (SNPs) which might not be the best way to reduce the features space. The technical team is not sure about the uniqueness of the method used to reduce the dimension of the somatic mutations.
&#x1F534;_**(Question: Does the audience know what the acronym SNP stands for? Full spellings have been added. )**_

### Components:	
&#x1F534;_**(Question: Will all three links point to the same asset in MoDaC? Yes)**_
* Untrained model: 
  * The untrained neural network model is defined in the model topology file [p1b2.model.json](https://modac.cancer.gov/searchTab?dme_data_id=). 
* Data:
  * The processed training and test data are in [ToDo: MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=).
* Trained Model:
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference [p1b2.model.h5](https://modac.cancer.gov/searchTab?dme_data_id=).

### Technical Details:
Refer to this [README](./Pilot1/P1B2/README.md).
