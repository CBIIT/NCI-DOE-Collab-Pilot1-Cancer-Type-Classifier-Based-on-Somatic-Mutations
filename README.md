# NCI-DOE-Collab-Pilot1-Mutation_Classifier

### Description:
The Mutation Classifier capability (Pilot 1 Benchmark 2, also known as P1B2) shows how to build a deep learning network that can classify the cancer type.

### User Community:	
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The current code makes heavy use of CANDLE Application Programming Interfaces (APIs). It can be used by a data scientist experienced in python and the domain.

### Uniqueness:	
Autoencoder are not the only method for dimensionality reduction. Other techniques like principal component analysis, t-distributed Stochastic Neighbor Embedding (tSNE), and Uniform Manifold Approximation and Projection (UMAP) are popular for molecular data. For high dimensional input vectors autoencoder can be beneficial, but this needs to be investigated.

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
