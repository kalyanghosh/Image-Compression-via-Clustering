%Topics in Data Science Project1
%Project Name:Image Compression via Clustering
%Project Members:
%Kalyan Ghosh(kghosh)
%Bhargav Mysore(bmysore)
%1.Loading the image(View of the EB):
I=imread('ncsu.jpg');
%figure(1),
%imshow(I);
%title('Original Input RGB Image');

%Converting to gray image from RGB:
rgbI=rgb2gray(I);
%figure(2),
%imshow(rgbI);
%title('Gray Image');



%Processinng the image to get the correct dimensions
%Getting the dimension of the gray image:
[M,N]=size(rgbI);

%Reshaping the image to dimensions M=512,N=2*512:
J=imresize(rgbI,[512,2*512]);
[m,n]=size(J);


%Selecting a good subset of the image:
p=[100:300];
q=[420:600];

subset=J(p,q,:);
%figure(3),
%imshow(subset);
%title('Part of Original Image');

%Choosing the Patch Size,P=2,resulting in 4 pixels per patch:
hor_dim=2;
ver_dim=2;


%Initializing xnew parameterized matrix:
xnew=zeros(hor_dim*ver_dim,(m*n)/(hor_dim*ver_dim));
[a,b]=size(xnew);

k=1;
%Looping through the image and storing the patches in rows
%of xnew.Note:Each Row of xnew stores the pixel of a patch and there are
%MN/P^2 number of such patches
for i=1:hor_dim:m
    for j=1:ver_dim:n
        patch=J(i:i+hor_dim-1,j:j+ver_dim-1);
        xnew(:,k)=patch(:);
        k=k+1;
    end
end
 

xtranspose=xnew(:);

%Applying KMeans Clustering on the patches of the image

[idx,C]=kmeans(xtranspose,16);




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


figure(4),
imshow(J);
title('Vector Quantized whole image');

quantized_image=J(p,q,:);
figure(5),
imshow(quantized_image)
title('Part of quantized image');