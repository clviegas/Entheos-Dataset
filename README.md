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

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{viegas2021entheos,
    	Address = {Online},
    	Author = {Viegas, Carla and Alikhani, Malihe},
    	Booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
    	Date-Added = {2021-08-04 14:21:26 -0400},
    	Date-Modified = {2021-08-04 14:22:05 -0400},
    	Doi = {10.18653/v1/2021.findings-acl.180},
    	Month = aug,
    	Pages = {2047--2060},
    	Publisher = {Association for Computational Linguistics},
    	Title = {Entheos: {A} Multimodal Dataset for Studying Enthusiasm},
    	Url = {https://aclanthology.org/2021.findings-acl.180},
    	Year = {2021},
    	Bdsk-Url-1 = {https://aclanthology.org/2021.findings-acl.180},
    	Bdsk-Url-2 = {https://doi.org/10.18653/v1/2021.findings-acl.180}}


## Acknowledgements

This work was supported by TalkMeUp Inc.
