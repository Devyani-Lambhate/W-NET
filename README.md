# W-NET
A Deep Network for Simultaneous Identification of Gulf Stream and Rings from Concurrent Satellite Images of Sea Surface Temperature and Height

#Libraries required
1. pytorch 
2. Matplotlib
3. OpenCv
4. Pillow

# Download SST and SSH data
## SST data
 1. Mount the podaac drive on your local machine using: https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA
 2. Run Download_data/SST_Poodac.py to download the nc files
 3. Run Download_data/SST_nc_to_colormap.py to convert  Net-CDF files to images.
 
## SSH data
 1. Download all the NC files from year 2014 to 2018 from the below link- 
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview
 2. Run Download_data/SSH_nc_to_colormap.py
 
#Training
For training the model Run train.py 
    train.py uses load_data.py to load the data. loss.py file constains differnt loss function used for training. Models contains different models used for ablation studies.
 
#Test
 4. Run Test/test.py
    test.py gives the final test accuracy and it saves the overlapping ground-truth and perdicted images for a better visual inference. It also finds and prints the number of detected warm and cold eddies.
    
#Further Evaluation
thinning.py performs the thinning operation on the gulf stream.
Code for further evaluation is saved in Model Evaluation methods. 
Hausdorff_distance.py computes the Hausdorff distance, Mean curve distance and Median curve distance.
path_length.py computs the path length difference between the ground truth and the predicted gulf stream centerline. 



