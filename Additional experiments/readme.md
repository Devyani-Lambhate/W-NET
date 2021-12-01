These additional experiments are not the part of the W-Net paper

## Canny Edge detection
Canny edge detection is performed separately on SST and SSH images but the result was not good enough using this technique. This also implies that simple edge detection techniques are not sufficient for complex features like Gulf stream and eddies.

## Remove false eddies
We also tried to further clean the data provided by Jennifer Clark by removing some flase eddies. 
False eddies are the eddies wrongly labeld as eddies. Our definition of false eddy is as follows-
An eddy that is not present in consecutive four images is the false eddy.

## Dijectra's Algorithm for Gulf Stream centerline
We tried using a nodification of Dijectra's algorithm to remove the meanders but we were unsuccessful in this because of some algorithmic limitations. 
