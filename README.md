# Entheos-Dataset

Source code for "Entheos: A Multimodal Dataset for Studying Enthusiasm" (Carla Viegas and Malihe Alikhani - ACL 2021).


## Dependencies

- Pytorch (follow instructions on [official website](https://pytorch.org/get-started/locally/)   
- Skorch    
`pip install -U skorch`    
- Python 3.6 or higher   

Other dependencies can be found in the Pipfile.

##  Usage

In the bash script run.sh you can find commands to start training the best unimodal and best multimodal models. If you want to train all of them simply run:

$ bash run.sh

If you want to train a model with different modalities, create a new config file similar to the existing ones, changing the "feature" and "modalities" entry to what you desire. In the CSVFeatures folder are all features extracted from the different modalities. This allows you to easily choose the modalities you want to use by adding the path to the CSV files in the config files.
