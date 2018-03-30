******************************************************************************************
VECTOR QUANTIZATION VIA K-MEANS CLUSTERING:

This entire code base,images and explanation can be found in this repository



******************************************************************************************







1. Loading an Image:
	
1.1
In this project, we have used the image of NCSU Engineering Building campus and  as shown below:

The image was loaded with below MATLAB code (imread):

%1.Loading the image(View of the EB):
I=imread('ncsu.jpg');
figure(1),
imshow(I);
title('Original Input RGB Image');

![NCSU IMAGE](https://github.com/kalyanghosh/Image-Compression-via-Clustering/blob/master/ncsu.jpg)

*******************************************************************************************

1.2
The input RGB image is then converted into a gray scale image by using the below MATLAB code (rgb2gray):

%Converting to gray image from RGB:
rgbI=rgb2gray(I);
figure(2),
imshow(rgbI);
title('Gray Image');

******************************************************************************************
1.3
Now ,we are resizing the image into appropriate dimensions for mathematical simplicity,where we are choosing min{M,N} to be a power of 2 and and max{M,N} is a multiple of that power of 2 . In our case we have taken M=512 and N=2X512.

We are resizing the rgb image using the below MATLAB code:

%Processinng the image to get the correct dimensions
%Getting the dimension of the gray image:
[M,N]=size(rgbI);
 
%Reshaping the image to dimensions M=512,N=2*512:
J=imresize(rgbI,[512,2*512]);
[m,n]=size(J);
***********************************************************************************************
1.4
Now we select a subset of the M X N image that looks nice.This subset of the image will be compared with its version after Vector Quantization.
The subset of the image is selected using the below MATLAB code:

%Selecting a good subset of the image:
p=[100:300];
q=[420:600];
 
subset=J(p,q,:);
figure(3),
imshow(subset);
title('Part of Original Image');

*********************************************************************************************

2. Clustering and Vector Quantization:

2.1
Now after the image preprocessing is done in the first step, we define a "patch" in the image with dimensions PxP, where P=2. This value of will work well for the dimensions of the above M X N image since they are also powers of 2.
In total,there would be (MxN)/P^2 patches. We then cluster these patches into 16 clusters using k-means clustering algorithm. We use the kmeans command in MATLAB for this. 
The value of the number of clusters (C) is chosen by choosing the value of R as 1, where R (rate) as 1. This means, each pixel is represented by RP2 number of bits. Total number of clusters C is obtained using, C =2(RP^2). 
When k-means algorithm is applied, each patch in the image would be assigned to a particular cluster. Replace each patch by the values of the cluster to which it belongs. 
We observe that as the number of patches decreases the quality of the compressed image deteriorates(at cluster=1, image is totally destroyed). 
We now observe that the whole image is now represented using only those values in the cluster. Hence, we will be able to represent the whole image in lesser number of bits than before.
Thus, image compression has been achieved.

Below is the plot of part of the image and its quantized version:
P=2,R=1 C=16
![NCSU IMAGE](https://github.com/kalyanghosh/Image-Compression-via-Clustering/blob/master/quantized_ncsu.png)

*******************************************************************************************

3 Rate vs. Distortion

 In this part, we vary the R(rate) value from 0 to 1. As we vary the R value, number of clusters C also changes because C=2(R*P^2). For each value of R (and thereby C) we do the clustering and the Vector Quantization again. For each of these R we calculate the distortion between the original image and the quantized image using the formula:
                                             
Where M is the number of rows in the image, N is the number of columns in the image, x is the pixel value of original image and y is the pixel value of the quantized image.
P=2


We observe that as we increase the rate R, the distortion is monotonically decreasing. This happens because as we increase R keeping patch size constant, the number of clusters C increase.


![Rate vs Distortion](https://github.com/kalyanghosh/Image-Compression-via-Clustering/blob/master/Rate_vs_Distortion.png)
Distortion is the measure of how close the quantized image is to the original image.
As the more and more clusters are used to represent the same image, the reconstructed image goes closer and closer to the original image and hence the distortion is less.
Basically we are using more number of values we are using to represent the same image, so the values will be more closer to the values of original image.
The trade-off here is that as R increases the number of bits required to represent also increases.

Note: We need to choose R values such that R*P2 should be an integer. Here we have taken R=[0.25,0.5,0.75,1.0]

***************************************************************************************************

4.Patch Size:

In this part, we vary the patch size of the patches.Earlier we had used P=2 to generate a 2x2 patch size.Now, in this part we use some value of P that is a powe of 2.For our example we have used P=4 to generate a patch size of 4x4 and plotted the Rate vs Distortion plot.

We now compare the Rate vs Distortion plot for P=2 and P=4 to see that if the RD tradeoff improved or degraded.

Plot 1: R=[0.25 0.5 1] ,P=2  and Plot 2 :R=[0.125 0.25 0.5] and P=4.

The values of R are so chosen that the value 2^k where(k=RP^2) returns integral values(Clusters).

Explanation:

We see, that with increasing values of R and P,the Distortion decreases.This can be explained by the below points:


When P=2 and R is [0.25 0.5 1],the values of k=RP^2 are (1,2,4) and consequently the values of 2^k  are (2,4,16) clusters
When P=4 and R=[0.125 0.25 0.5],the values of k=RP^2 are(2,4,8) and consequently the values of 2^k are (4,16,256) clusters.
So,we see that with increasing patch size in powers of 2,the value of 2^k i.e the numbers of clusters also increases.
We know with increasing Patch Size,the distortion value goes up ,because we are approximating a larger portion of the image.But on the other hand,with increasing value of patch size,the cluster number also goes up and we know that  increasing the number of clusters decreases the distortion because we are dividingthe image into more number of clusters and hence decreasing the level of approximation of the image.
In the above example,for P=2 and R=0.25 and 0.5,the values of Distortion are:110 and 68 respectively and the cluster numbers are 2 and 4 respectively.Similarly, for P=4 and R=0.25 and R=0.5,the values of Distortion are 40 and 35 respectively and the cluster numbers are 16 and 256 respectively.
It is evident from the above values that for P=4,the number of clusters is much larger that the number of clusters for P=2 for same value of R.So,even though P=4 is greater that P=2 which should have increased the distortion,we see that the distortion for P=4 is less than P=2 for same values of R.
So,it is evident from the above discussion that,the number of clusters dominates the increase of patch size in reducing the distortion.


******************************************************************************************************


 Better Compression

Till now we have considered that all the clusters have equal probability of appearing in the image. However in reality, some clusters appear more often than the others. Therefore, in practical applications we usually represent the clusters which appear frequently with lesser number of bits. Information theory tells us that we can get an average coding length of  if we take into consideration that some symbols appear more often than others.
In this part we compare the average coding rate for uniform probability and the actual probability.


We have calculated  H and R(actual coding rate) for different values of R. We see from the first entry that we could save 1-0.8723=0.1277 bits per patch if we had not assigned equal coding length for every patch. Also the average number of bits required for each patch is 3.4893 instead of 4.
Hence, in practical applications we observe that always the frequency of appearance of one symbol will be more than others. We can decrease the coding length by representing it using lesser number of bits. We assign more bits for representing symbols which appear less frequently.

********************************************************************************************************

Explanation:

Here we run the k means clustering algorithm on a training image,collect the cluster means and then test it on a testing image.

Our findings can be discussed in the below points.

Here ,first of all we run the k means clustering algorithm for P=2,R=1 and k=16 on our training image.
After runnnig the kmeans .we store the 16 cluster centroids.
The quantized version of the training image is shown as below.

The quantized version of the training image is shown as below.
We find that error is (Error of Training Image for R=1 and P=2  and k=16 is =25.431574
We then take a new test image.The original version of the test image is shown as below: 


We then run k means clustering on the test image with R=1,patch size P=2(same as in training process) and k=16(same as in training process) and replace the representation patch with values as received in the training process.

*TEST_IMAGE:
![Test Image] (https://github.com/kalyanghosh/Image-Compression-via-Clustering/blob/master/hunt.png)

*VECTOR QUANTIZED TEST_IMAGE:
![Test Image] (https://github.com/kalyanghosh/Image-Compression-via-Clustering/blob/master/quantized_hunt.png)
*******************************************************************************************************

