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
    ~cGrid() { } 

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
	void addValue(int ix, int iy, int iz, float value){
    	cVector3Idx idx(ix,iy,iz);
    	if( (idx.x>=0)&&(idx.x<data->size[0]) && (idx.y>=0)&&(idx.y<data->size[1]) && (idx.z>=0)&&(idx.z<data->size[2]) ){
			THFloatTensor_set3d(data, idx.x,idx.y,idx.z,THFloatTensor_get3d(data, idx.x,idx.y,idx.z)+value);
    	}else{
    		return;
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
	float extrapolate(const cVector3& location, float value) 
    { 
        float gx, gy, gz, tx, ty, tz;
        unsigned gxi, gyi, gzi; 
        // remap point coordinates to grid coordinates
        cVector3Idx gri = getTensorIndexes(location);
        
        tx = location.v[0] - gri.x; 
        ty = location.v[1] - gri.y; 
        tz = location.v[2] - gri.z; 
       
        const float c000 = (1-tx)*(1-ty)*(1-tz);
		const float c100 = tx*(1-ty)*(1-tz);
		const float c010 = (1-tx)*ty*(1-tz);
		const float c110 = tx*ty*(1-tz);

		const float c001 = (1-tx)*(1-ty)*tz;
		const float c101 = tx*(1-ty)*tz;
		const float c011 = (1-tx)*ty*tz;
		const float c111 = tx*ty*tz;
		

		addValue(gri.x, gri.y, gri.z, c000*value);
		addValue(gri.x+1, gri.y, gri.z, c100*value);
    	addValue(gri.x, gri.y+1, gri.z, c010*value);
		addValue(gri.x+1, gri.y+1, gri.z, c110*value);
		addValue(gri.x, gri.y, gri.z+1, c001*value);
        addValue(gri.x+1, gri.y, gri.z+1, c101*value);
		addValue(gri.x, gri.y+1, gri.z+1, c011*value);
        addValue(gri.x+1, gri.y+1, gri.z+1, c111*value);

    } 

	cVector3 grad2fwd(int ix, int iy, int iz) 
    { 
        const float c000 = getValue(ix, iy, iz);//data[IX(gei, gyi, gzi)]; 
        const float c100 = getValue(ix+1, iy, iz);//data[IX(gxi + 1, gyi, gzi)]; 
        const float c010 = getValue(ix, iy+1, iz);//data[IX(gxi, gyi + 1, gzi)]; 
        const float c001 = getValue(ix, iy, iz+1);//data[IX(gxi + 1, gyi + 1, gzi)]; 

		return cVector3(c100-c000, c010-c000, c001-c000);
    } 

	float grad2bwd(int ix, int iy, int iz) 
    { 
        const float c000 = getValue(ix, iy, iz);//data[IX(gei, gyi, gzi)]; 
        const float c100 = getValue(ix+1, iy, iz);//data[IX(gxi + 1, gyi, gzi)]; 
        const float c010 = getValue(ix, iy+1, iz);//data[IX(gxi, gyi + 1, gzi)]; 
        const float c001 = getValue(ix, iy, iz+1);//data[IX(gxi + 1, gyi + 1, gzi)]; 
		const float cm100 = getValue(ix-1, iy, iz);//data[IX(gxi + 1, gyi, gzi)]; 
        const float c0m10 = getValue(ix, iy-1, iz);//data[IX(gxi, gyi + 1, gzi)]; 
        const float c00m1 = getValue(ix, iy, iz-1);//data[IX(gxi + 1, gyi + 1, gzi)]; 

		return 2.0*(c000-c100 + c000-c010 + c000-c001) + 2.0*(c000 - cm100 + c000 - c0m10 + c000 - c00m1);
    } 
}; 

extern "C"{	 
void interpolateTensor(THFloatTensor *src, THFloatTensor *dst){
	cVector3 dst_r, src_r;
	double scale_x = (float(src->size[0])/float(dst->size[0]));
	double scale_y = (float(src->size[1])/float(dst->size[1]));
	double scale_z = (float(src->size[2])/float(dst->size[2]));
	// std::cout<<scale_x<<" "<<dst->size[0]<<std::endl;
	cGrid src_grid(src);

	for(int ix=0; ix<dst->size[0];ix++){
		for(int iy=0; iy<dst->size[1];iy++){
			for(int iz=0; iz<dst->size[2];iz++){
				// std::cout<<"("<<ix<<","<<iy<<","<<iz<<")"<<std::endl;
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

void extrapolateTensor(THFloatTensor *src, THFloatTensor *dst){
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
				src_grid.extrapolate(src_r, THFloatTensor_get3d(dst, ix, iy, iz));
			}
		}
	}
}

void forwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst){
	cGrid src_grid(src);
	cVector3 grad;
	float res = 0.0; 
	for(int ix=0; ix<src->size[0];ix++){
		for(int iy=0; iy<src->size[1];iy++){
			for(int iz=0; iz<src->size[2];iz++){
				grad = src_grid.grad2fwd(ix, iy, iz);
				res += grad.norm2();
			}
		}
	}
	THFloatTensor_set1d(dst, 0, res);
}

void backwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst){
	cGrid src_grid(src);
	for(int ix=0; ix<src->size[0];ix++){
		for(int iy=0; iy<src->size[1];iy++){
			for(int iz=0; iz<src->size[2];iz++){
				THFloatTensor_set3d(dst, ix, iy, iz, THFloatTensor_get3d(dst, ix, iy, iz) + src_grid.grad2bwd(ix, iy, iz));
			}
		}
	}
}

}