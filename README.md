
# FR-UNet

 
## Prerequisites
 

 
Download our repo:

## Datasets processing
Choose a path to create a folder with the dataset name and download datasets [DRIVE](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0),[CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/),[STARE](https://cecas.clemson.edu/~ahoover/stare/probing/index.html),[CHUAC](https://figshare.com/s/4d24cf3d14bc901a94bf), and [DCA1](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html). Type this in terminal to run the data_process.py file
 
```
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```
 
## Training
Type this in terminal to run the train.py file
 
```
python train.py -dp DATASET_PATH
```
## Test
Type this in terminal to run the test.py file
 
```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH (--sliding -ps 48 --stride 44)
```
We have prepared the pre-trained models for both datasets in the folder 'pretrained_weights'. To replicate the results in the paper, directly run the following commands
```
python test.py -dp DATASET_PATH -wp pretrained_weights/DATASET_NAME
```

## Export

```
python onnx_export.py --export_path outputs/onnx/frunet_fp.onnx
```

quan export

 
## License
