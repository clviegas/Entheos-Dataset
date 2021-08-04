#usr/bin/bash

# Binary
python $src_folder/gridSearch_skorch.py --JSON_path ./Config/UM_bert.json --num_classes 2 --model NeuralNet --fusion concat --scoring balanced_accuracy
python $src_folder/gridSearch_skorch.py --JSON_path ./Config/MM_all.json --num_classes 2 --model NeuralNet --fusion concat --scoring balanced_accuracy



# Multiclass
python $src_folder/gridSearch_skorch.py --JSON_path ./Config/UM_bert.json --num_classes 3 --model NeuralNet --fusion concat --scoring balanced_accuracy
python $src_folder/gridSearch_skorch.py --JSON_path ./Config/MM_all.json --num_classes 3 --model NeuralNet --fusion concat --scoring balanced_accuracy
