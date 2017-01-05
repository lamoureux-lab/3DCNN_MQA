#include "TensorMathUtils.h"
#include "cVector3.hpp"
#include <math.h>

float bilinear(const float tx, const float ty, const float c00, const float c10, const float c01, const float c11)
{ 
	float a = c00 * (1 - tx) + c10 * tx;
	float b = c01 * (1 - tx) + c11 * tx;
	return a * (1 - ty) + b * ty;
}


class cGrid
{
	public:
		THFloatTensor *data;
		
	cGrid(THFloatTensor *tensor)
	{
		data = tensor;
		if( tensor->nDimension != 3){
			std::cout<<"Non-3D tensor, cannot construct Grid\n";
		}
		
    } 
    ~cGrid() { if (data) delete [] data; } 

    cVector3Idx getTensorIndexes(cVector3 r){
    	// unsigned ix = fmax(0,fmin(floor(r.v[0]),data->size[1]-1));
    	// unsigned iy = fmax(0,fmin(floor(r.v[1]),data->size[2]-1));
    	// unsigned iz = fmax(0,fmin(floor(r.v[2]),data->size[3]-1));

    	unsigned ix = int(r.v[0]);
    	unsigned iy = int(r.v[1]);
    	unsigned iz = int(r.v[2]);
    
    	return cVector3Idx(ix,iy,iz);
    }
    
    float getValue(cVector3 r){
    	cVector3Idx idx;
    	idx = getTensorIndexes(r);
    	if( (idx.x>=0)&&(idx.x<data->size[0]) && (idx.y>=0)&&(idx.y<data->size[1]) && (idx.z>=0)&&(idx.z<data->size[2]) ){
    		return THFloatTensor_get3d(data, idx.x,idx.y,idx.z);
    	}else{
    		return 0.0;
    	}
    }

    float getValue(int ix, int iy, int iz){
    	cVector3Idx idx(ix,iy,iz);
    	if( (idx.x>=0)&&(idx.x<data->size[0]) && (idx.y>=0)&&(idx.y<data->size[1]) && (idx.z>=0)&&(idx.z<data->size[2]) ){
    		return THFloatTensor_get3d(data, idx.x,idx.y,idx.z);
    	}else{
    		return 0.0;
    	}
    }
    
    float interpolate(const cVector3& location) 
    { 
        float gx, gy, gz, tx, ty, tz;
        unsigned gxi, gyi, gzi; 
        // remap point coordinates to grid coordinates
        cVector3Idx gri = getTensorIndexes(location);
        
        tx = location.v[0] - gri.x; 
        ty = location.v[1] - gri.y; 
        tz = location.v[2] - gri.z; 
        // const float c000 = THFloatTensor_get3d(data, gri.x, gri.y, gri.z);//data[IX(gei, gyi, gzi)]; 
        // const float c100 = THFloatTensor_get3d(data, gri.x+1, gri.y, gri.z);//data[IX(gxi + 1, gyi, gzi)]; 
        // const float c010 = THFloatTensor_get3d(data, gri.x, gri.y+1, gri.z);//data[IX(gxi, gyi + 1, gzi)]; 
        // const float c110 = THFloatTensor_get3d(data, gri.x+1, gri.y+1, gri.z);//data[IX(gxi + 1, gyi + 1, gzi)]; 
        // const float c001 = THFloatTensor_get3d(data, gri.x, gri.y, gri.z+1);//data[IX(gxi, gyi, gzi + 1)]; 
        // const float c101 = THFloatTensor_get3d(data, gri.x+1, gri.y, gri.z+1);//data[IX(gxi + 1, gyi, gzi + 1)]; 
        // const float c011 = THFloatTensor_get3d(data, gri.x, gri.y+1, gri.z+1);//data[IX(gxi, gyi + 1, gzi + 1)]; 
        // const float c111 = THFloatTensor_get3d(data, gri.x+1, gri.y+1, gri.z+1);//data[IX(gxi + 1, gyi + 1, gzi + 1)]; 

        const float c000 = getValue(gri.x, gri.y, gri.z);//data[IX(gei, gyi, gzi)]; 
        const float c100 = getValue(gri.x+1, gri.y, gri.z);//data[IX(gxi + 1, gyi, gzi)]; 
        const float c010 = getValue(gri.x, gri.y+1, gri.z);//data[IX(gxi, gyi + 1, gzi)]; 
        const float c110 = getValue(gri.x+1, gri.y+1, gri.z);//data[IX(gxi + 1, gyi + 1, gzi)]; 
        const float c001 = getValue(gri.x, gri.y, gri.z+1);//data[IX(gxi, gyi, gzi + 1)]; 
        const float c101 = getValue(gri.x+1, gri.y, gri.z+1);//data[IX(gxi + 1, gyi, gzi + 1)]; 
        const float c011 = getValue(gri.x, gri.y+1, gri.z+1);//data[IX(gxi, gyi + 1, gzi + 1)]; 
        const float c111 = getValue(gri.x+1, gri.y+1, gri.z+1);//data[IX(gxi + 1, gyi + 1, gzi + 1)]; 

        float e = bilinear(tx, ty, c000, c100, c010, c110); 
        float f = bilinear(tx, ty, c001, c101, c011, c111); 
        return e * ( 1 - tz) + f * tz; 

    } 
}; 
	 
void interpolateTensor(THFloatTensor *src, THFloatTensor *dst){
	cVector3 dst_r, src_r;
	double scale_x = (float(src->size[0])/float(dst->size[0]));
	double scale_y = (float(src->size[1])/float(dst->size[1]));
	double scale_z = (float(src->size[2])/float(dst->size[2]));
	cGrid src_grid(src);

	for(int ix=0; ix<dst->size[0];ix++){
		for(int iy=0; iy<dst->size[1];iy++){
			for(int iz=0; iz<dst->size[2];iz++){
				dst_r = cVector3(ix,iy,iz);
				src_r.v[0] = dst_r.v[0]*scale_x;
				src_r.v[1] = dst_r.v[1]*scale_y;
				src_r.v[2] = dst_r.v[2]*scale_z;
				float interpolated_value = src_grid.interpolate(src_r);
				THFloatTensor_set3d(dst, ix, iy, iz, interpolated_value);
			}
		}
	}
}