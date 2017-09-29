%Topics in Data Science Project1
%Project Name:Image Compression via Clustering
%Project Members:
%Kalyan Ghosh(kghosh)
%Bhargav(bmysore)
close all;

clc;
%1.Loading the image(View of the EB):
I=imread('ncsu.jpg');
% figure(1),
% imshow(I);
% title('Original Input RGB Image');

%Converting to gray image from RGB:
rgbI=rgb2gray(I);
% figure(2),
% imshow(rgbI);
% title('Gray Image');



%Processinng the image to get the correct dimensions
%Getting the dimension of the gray image:
[M,N]=size(rgbI);

%Reshaping the image to dimensions M=512,N=2*512:
J=imresize(rgbI,[512,2*512]);
J1=imresize(rgbI,[512,2*512]);
[m,n]=size(J);


%Selecting a good subset of the image:
p=[100:300];
q=[420:600];

subset=J(p,q,:);
% figure(3),
% imshow(subset);
% title('Part of Original Image');

%Choosing the Patch Size,P=2,resulting in 4 pixels per patch:
hor_dim=2;
ver_dim=2;

%Initializing x_new parameterized matrix:
xnew=zeros(hor_dim*ver_dim,(m*n)/(hor_dim*ver_dim));
[a,b]=size(xnew);

k=1;
%Looping through the image and storing the patches in rows
%of xnew.Note:Each Row of xnew stores the pixel of a patche and there are
%N number of such patches
for i=1:hor_dim:m
    for j=1:ver_dim:n
        patch=J(i:i+hor_dim-1,j:j+ver_dim-1);
        xnew(:,k)=patch(:);
        k=k+1;
    end
end
    
xtranspose=xnew(:);
%disp(length(xtranspose))
%disp(xnew);
%Applying KMeans Clustering on the 


    
    [idx,C]=kmeans(xtranspose,16);
    codebook=C(:);

    %Averaging along each row of the Code Book
    %CodeBook=mean(C,2);
    %disp(C)
    %disp(idx);
    %disp(CodeBook);

    %Now running Vector Quantization
    %Replacing the patch with the cluster centroid to which it belongs

   a=1;
   for i=1:hor_dim:m
      for j=1:ver_dim:n
        patch=J(i:i+hor_dim-1,j:j+ver_dim-1);
        cluster_number=idx(a);
        cluster_mean=C(cluster_number);
        J(i:i+hor_dim-1,j:j+ver_dim-1)=cluster_mean;
        a=a+(hor_dim*ver_dim);
      end
  end
   err=J1-J;
   error=((1/(m*n))*(sum(sum(err.^2))));
   fprintf('Error of Training Data for R=1 and P=2 is =%f \n',error)
   imshow(J)
   
   %*********************************************************************
   
I2=imread('hunt.png');
rgbI2=rgb2gray(I2);
J2=imresize(rgbI2,[512,2*512]);
J2_temp=imresize(rgbI2,[512,2*512]);
[m1,n1]=size(J2);

p1=[100:300];
q1=[420:600];

xnew1=zeros(hor_dim*ver_dim,(m*n)/(hor_dim*ver_dim));
[a1,b1]=size(xnew1);

%Running K Means on the new image and replacing the Cluster Means with the
%entries we have from the training data


k=1;
%Looping through the image and storing the patches in rows
%of xnew.Note:Each Row of xnew stores the pixel of a patch and there are
%MN/P^2 number of such patches
for i=1:hor_dim:m1
    for j=1:ver_dim:n1
        patch=J2(i:i+hor_dim-1,j:j+ver_dim-1);
        xnew1(:,k)=patch(:);
        k=k+1;
    end
end
 
%disp(xnew)
xtranspose1=xnew1(:);
%disp(length(xtranspose1))

[idx1]=kmeans(xtranspose1,16);

a=1;
for i=1:hor_dim:m
    for j=1:ver_dim:n
        patch=J2(i:i+hor_dim-1,j:j+ver_dim-1);
        cluster_number=idx1(a);
        cluster_mean=codebook(cluster_number);
        J2(i:i+hor_dim-1,j:j+ver_dim-1)=cluster_mean;
        a=a+(hor_dim*ver_dim);
    end
end
err1=J2-J2_temp;
   error=((1/(m*n))*(sum(sum(err1.^2))));
   fprintf('Error of Test Data for R=1 and P=2 is =%f \n',error)
imshow(I2)

   

