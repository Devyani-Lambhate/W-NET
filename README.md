# W-NET
From the literature we found that Sea Surface Temperature (SST) and Sea Surface Height (SSH) are the primary oceanographic variables that characterize the gulf stream and rings. We designed a network(W-Net) that could use SST and SSH data for automated and simultaneous identification of these synoptic ocean features.

Main Libraries required
1. pytorch 
2. Matplotlib
3. OpenCv
4. Pillow

## Download SST and SSH data
Our dataset consists of Sea Surface Temperature (SST) maps, Sea Surface Height (SSH) maps [Fig.1(b)], and the manual annotations of gulf stream and eddies by an expert. Our focus area is the region bounded by 85◦W to 55◦W and 20◦N to 45◦N. You can download the SST and SSH data using the steps described in the next section, but the maunal annotations of gulf stream and eddies are not publicaly available so we will skip that here. 

### SST data
We used the Level 4 Operational Sea Surface Temperature and Sea Ice Analysis (OSTIA) product as our SST input. This data is available at 5.5km resolution. We also tried 1km SST data but did not get any improvement in the network's performance.

 link for OSTIA data : https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA/
 link for 1km SST data : https://podaac tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/L4/GLOB/JPL_OUROCEAN/G1SST/
 
 1. Mount the podaac drive on your local machine using: 
    sudo mount.davfs web_link /your_directory
 2. Run Download_data/SST_Poodac.py to download the Net-CDF files
 3. Run Download_data/SST_nc_to_colormap.py to convert Net-CDF files to images
 
### SSH data
We used the gridded product of sea-level anomaly provided by copernicus marine environmental monitoring service (CMEMS) as our SSH
input. This data is available at 27km resolution. We need to manually download the SSH data from the link below-

 1. Download all the NC files from year 2014 to 2018 from the below link- 
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview
 2. Run Download_data/SSH_nc_to_colormap.py to convert Net-CDF files to images
 
## Training
To train the model run train.py </br>
train.py uses load_data.py to load the data. </br>
loss.py file constains differnt loss function used for training. One can add custom loss functions here and use these new losses to train the model. </br>
train.py calls w_net_model_r_1.py as it contains the main W-Net architecture. 

There are a few thing that user can specify in the train.py to perform several experiments
1. Number of classes and labels of classes. The network can be trained on all four labels (Warm eddies, Cold eddies, Gulf Stream and background) or it can be trained on a subset of these labels.
2. Loss function (crossentropy, Dice or a mix of both)
3. Type of model (W-Net, Res-W-Net, Y-Net, U-Net-SST, U-Net-SSH)
4. Data (One can use full data, winter months data or summer months data)
The train.py saves the model in w-net.pth, which can be used for testing and further evaluation.
 
## Test
For testing run Test/test.py
    test.py gives the final test accuracy and it saves the overlapping ground-truth and perdicted images for a better visual inference. It also finds and prints the number of detected warm and cold eddies.
    
## Further Evaluation
thinning.py performs the thinning operation on the gulf stream.
Code for further evaluation is saved in Model Evaluation methods. 
Hausdorff_distance.py computes the Hausdorff distance, Mean curve distance and Median curve distance.
path_length.py computs the path length difference between the ground truth and the predicted gulf stream centerline. 



