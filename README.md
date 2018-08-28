## Personalized Photo Recommendation
This repository includes work for the **Aesthetic Features for Personalized Photo Recommendation**.

### Data
You need to save your data in the folder `data/`. If you only have one dataset, you can save under the folder with name 
'sample/'. If you have multiple datasets, you can save multiple them in separate folders with any name format. 

Under each folder (for one datset), you have a folder named 'validation/' for parameter tuning, 'test/' for 
evaluation, and `photos` to save all photos to be used. Under each of the folder, there will be `matrix_train.npz` for training and `matrix_test.npz` for testing.  
The photo data should be all in `.jpg` format. 

### Create Aesthetic Feature Embedding
#### Create Color Embedding
You can create color embedding for all your photos with the script below:
```
cd aesthetic_features
python create_color_histogram.py --data-folder data/
```

By default, the program will create RGB, HSV, and HLS embedding respectively for all photos.


#### Create Style Embedding
You can run style embedding for all your photos with the script below:
```
cd aesthetic_features
python create_style_embedding.py --data-folder data/
```

### Evaluate Models
You can evaluate with all the models on your data with the script below:
```
python model_evaluation.py --data-folder data/ --task test 
```