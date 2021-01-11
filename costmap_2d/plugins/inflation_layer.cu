#include <costmap_2d/cuda_inflation_layer.h>
#include <cmath>
#include <cstring>
#include <vector>

#define TPB 512

#define NO_INFORMATION 255
#define INSCRIBED_INFLATED_OBSTACLE 253
#define FREE_SPACE 0

using std::abs;
using std::memcpy;
using std::vector;
using costmap_2d::CellData;

__global__ void setCostFloodingInflateUnkown(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char *cachedCost_1D, unsigned int cacheSize, CellData *obstaclesArray, unsigned int obstaclesArray_count, unsigned int inflation_radius, int min_i, int min_j, int max_i, int max_j)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    int temp=id;

    //No need to handle those threads not assigned with tasks
    int obstacleNum=temp/((2*inflation_radius+1)*(2*inflation_radius+1));
    if(obstacleNum>obstaclesArray_count)
        return;
    
    temp%=(2*inflation_radius+1)*(2*inflation_radius+1);

    //No need to handle those threads too far away from obstacles
    int deltay=temp/(2*inflation_radius+1)-inflation_radius;
    int deltax=temp%(2*inflation_radius+1)-inflation_radius;
    if(deltax*deltax+deltay*deltay>inflation_radius*inflation_radius)
        return;

    //No need to handle those threads out of bounds
    int x=obstaclesArray[obstacleNum].src_x_+deltax;
    int y=obstaclesArray[obstacleNum].src_y_+deltay;
    if(x<min_i||x>=max_i||y<min_j||y>=max_j)
        return;
    
    unsigned int index=y*master_size_x+x;
    unsigned char cost=cachedCost_1D[abs(deltay)*cacheSize+abs(deltax)];
    unsigned char old_cost=master[index];

    if(old_cost==NO_INFORMATION&&cost>FREE_SPACE)
        master[index]=cost;
    else
        master[index]=cost>old_cost?cost:old_cost;
}

__global__ void setCostFloodingNoInflateUnkown(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char *cachedCost_1D, unsigned int cacheSize, CellData *obstaclesArray, unsigned int obstaclesArray_count, unsigned int inflation_radius, int min_i, int min_j, int max_i, int max_j)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    int temp=id;

    //No need to handle those threads not assigned with tasks
    int obstacleNum=temp/((2*inflation_radius+1)*(2*inflation_radius+1));
    if(obstacleNum>obstaclesArray_count)
        return;
    
    temp%=(2*inflation_radius+1)*(2*inflation_radius+1);

    //No need to handle those threads too far away from obstacles
    int deltay=temp/(2*inflation_radius+1)-inflation_radius;
    int deltax=temp%(2*inflation_radius+1)-inflation_radius;
    if(deltax*deltax+deltay*deltay>inflation_radius*inflation_radius)
        return;

    //No need to handle those threads out of bounds
    int x=obstaclesArray[obstacleNum].src_x_+deltax;
    int y=obstaclesArray[obstacleNum].src_y_+deltay;
    if(x<min_i||x>=max_i||y<min_j||y>=max_j)
        return;
    
    unsigned int index=y*master_size_x+x;
    unsigned char cost=cachedCost_1D[abs(deltay)*cacheSize+abs(deltax)];
    unsigned char old_cost=master[index];

    if(old_cost==NO_INFORMATION&&cost>INSCRIBED_INFLATED_OBSTACLE)
        master[index]=cost;
    else
        master[index]=cost>old_cost?cost:old_cost;
}

void costmap_2d::cuda::inflation_layer::setCostFlooding(unsigned char *master, unsigned long master_size_x, unsigned long master_size_y, unsigned char **cached_cost, const vector<CellData> &obstacles, unsigned int inflation_radius, bool inflate_unknown, int min_i, int min_j, int max_i, int max_j)
{
    unsigned int cacheSize=inflation_radius+2;
    
    if(obstacles.empty())
        return;
    
    //Compress the original 2D cached cost into 1D for more convenient cuda_memcpy
    unsigned char *cachedCost_1D=NULL;
    cudaMallocManaged(&cachedCost_1D, sizeof(unsigned char)*cacheSize*cacheSize, cudaMemAttachHost);
    for(int i=0;i<cacheSize;++i)
        memcpy(cachedCost_1D+cacheSize*i,cached_cost[i],cacheSize*sizeof(unsigned char));
    
    CellData *obstaclesArray=NULL;
    cudaMallocManaged(&obstaclesArray, sizeof(CellData)*obstacles.size(), cudaMemAttachHost);
    memcpy(obstaclesArray,&obstacles[0],obstacles.size()*sizeof(CellData));

    cudaStreamAttachMemAsync(NULL, master, 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, cachedCost_1D, 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, obstaclesArray, 0, cudaMemAttachGlobal);

    //2*inflation_radius+1 is actually inflation diameter, but we still had better pass raduis into kernel.
    unsigned long totalWorkload=obstacles.size()*(2*inflation_radius+1)*(2*inflation_radius+1);
    if(inflate_unknown)
        setCostFloodingInflateUnkown<<<(totalWorkload+TPB-1)/TPB,TPB>>>(master,master_size_x,master_size_y,cachedCost_1D,cacheSize,obstaclesArray,obstacles.size(),inflation_radius,min_i,min_j,max_i,max_j);
    else
        setCostFloodingNoInflateUnkown<<<(totalWorkload+TPB-1)/TPB,TPB>>>(master,master_size_x,master_size_y,cachedCost_1D,cacheSize,obstaclesArray,obstacles.size(),inflation_radius,min_i,min_j,max_i,max_j);
    
    cudaStreamAttachMemAsync(NULL, master, 0, cudaMemAttachHost);
    cudaStreamSynchronize(NULL);
    cudaFree(cachedCost_1D);
    cudaFree(obstaclesArray);
}
