# Python program for Dijkstra's single  
# source shortest path algorithm. The program is  
# for adjacency matrix representation of the graph 
  
# Library for INT_MAX 
import sys 
import cv2
import math
import numpy as np
import os
  
class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    def printSolution(self, dist,dest): 
        #print ("Vertex \tDistance from Source")
        return dist[dest]
        #for node in range(self.V): 
        #   print (node, "\t", dist[node]) 
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minDistance(self, dist, sptSet): 
  
        # Initilaize minimum distance for next node 
        min = float('inf')
  
        # Search not nearest vertex not in the  
        # shortest path tree 
        for v in range(self.V): 
            if dist[v] < min and sptSet[v] == False: 
                min = dist[v] 
                min_index = v 
  
        return min_index 
  
    # Funtion that implements Dijkstra's single source  
    # shortest path algorithm for a graph represented  
    # using adjacency matrix representation 
    def dijkstra(self, src, dest): 
  
        dist = [float('inf')] * self.V 
        dist[src] = 0
        sptSet = [False] * self.V 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minDistance(dist, sptSet) 
  
            # Put the minimum distance vertex in the  
            # shotest path tree 
            sptSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]: 
                        dist[v] = dist[u] + self.graph[u][v] 
  
        return self.printSolution(dist,dest)
  
# Driver program
#Our dir names
sst_dir = 'SST/'
ssh_dir = 'SSH/'
gt_dir = 'GT/' 
root=  'merged/test_data/'


path='merged/test_data'
sst_fnames = sorted(os.listdir(os.path.join(path,'SST')))
ssh_fnames = sorted(os.listdir(os.path.join(path, 'SSH')))

# Compare file names from GT folder to file names from RGB:
image_arr=[]
for gt_fname in sorted(os.listdir(os.path.join(path, 'GT'))):
    if gt_fname in sst_fnames and gt_fname in ssh_fnames:
         image_arr.append(gt_fname)

print(len(image_arr))

mean_path_difference=0
for image in image_arr:
    img=[]
    res=[]

    img.append(cv2.imread("Gulf Stream/True/"+image,0))
    img.append(cv2.imread("Gulf Stream/Pred/"+image,0))
   


    for k in range(2):
    
        img_cs1 = img[k]
       
        #print(np.unique(img_cs1))
        #print(img_cs2.shape)


        true=[]
        pred=[]

        for i in range(0,512):
            for j in range(0,410):
                if(img_cs1[j,i]!=0):
                    true.append([j,i])
                

        adj_matrix=[[0 for i in range(len(true))]for j in range(len(true))]
        i=0
        #print(true)
        for pix_pos_t in true:
            x_t,y_t=pix_pos_t

            j=0
            max_x=0
            y_at_max_x=0
            max_y=0
            x_at_max_y=0


            for pix_pos_t1 in true:
                
                x_t1,y_t1=pix_pos_t1
                if(x_t1>max_x):
                    max_x=x_t1
                    y_at_max_x=y_t1

                if(y_t1>max_y):
                    max_y=y_t1
                    x_at_max_y=x_t1
                
                d=math.sqrt((x_t1-x_t)**2+(y_t1-y_t)**2)
                #print(d)
                adj_matrix[i][j]=d
                j+=1
            i+=1

        #print(max_x,y_at_max_x,max_y,x_at_max_y)
        
        start=0
        end=0
        for m in range(len(true)):
            if(max_x==true[m][0] and y_at_max_x==true[m][1]):
                start=m
            elif(x_at_max_y==true[m][0] and max_y==true[m][1]):
                end=m


        #print(start,end)





        #print(adj_matrix)
        #adj=np.array(adj_matrix)
        #start=np.argmin(adj[:,1])
        #print(start)
        #end=np.argmax(adj[:,0])
        #print(end)
        #start=max

        #print(true)


        g=Graph(len(true))
        g.graph=adj_matrix
        ans=g.dijkstra(start,end)
        res.append(ans)
    mean_path_difference+=abs(res[0]-res[1])

print('mean_path_difference: ',mean_path_difference/len(image_arr))      
