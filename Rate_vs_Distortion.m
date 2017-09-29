%Topics in Data Science Project1
%Project Name:Image Compression via Clustering
%Project Members:
%Kalyan Ghosh(kghosh)
%Bhargav()
close all;
clear all;
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
R=[0.25,0.5,0.75,1.0];
for g=1:1:length(R)
    T=2^(floor(R(g)*hor_dim*ver_dim));
    [idx,C]=kmeans(xtranspose,T);

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
   error(g)=((1/(m*n))*(sum(sum(err.^2))));
end

plot(R,error);
xlabel(' Values of R ');ylabel( ' Distortion');
title('Rate vs Distortion Plot for P=2');

% figure(4),
% imshow(J);
% title('Vector Quantized whole image');
% 
% figure(5),
% quantized_image=J(p,q,:);
% imshow(quantized_image);
% title('Part of quantized image');
