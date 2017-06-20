#include "TensorMathUtils.h"
#include <iostream>
#include <math.h>
using namespace std;

float getDist(float x, float y, float z, float center){
	return sqrt( (x-center)*(x-center) + (y-center)*(y-center) + (z-center)*(z-center) );
}

float f1(int x, int y, int z){
	float center = 50;
	float r = 25; 
	if(getDist(x,y,z,center)< r){
		return 1.0;
	}
	return 0.0;
}

float f2(int x, int y, int z){
	float center = 25;
	float r = 12.5; 
	if(getDist(x,y,z,center)< r){
		return 1.0;
	}
	return 0.0;
}

void fillTensor(THFloatTensor *tensor, float (*f)(int, int, int)){
	for(int ix=0; ix<tensor->size[0];ix++){
		for(int iy=0; iy<tensor->size[1];iy++){
			for(int iz=0; iz<tensor->size[2];iz++){
				THFloatTensor_set3d(tensor, ix, iy, iz, f(ix, iy, iz));
	}}}
}

int main(void){

	cout<<"Tensor math utils test\n";
	int src_size = 10;
	int dst_size = 50;

	THFloatTensor *Tref = THFloatTensor_newWithSize3d(dst_size,dst_size,dst_size);
	fillTensor(Tref, f1);

	THFloatTensor *Tsrc = THFloatTensor_newWithSize3d(src_size,src_size,src_size);
	fillTensor(Tsrc, f2);

	THFloatTensor *Tdst = THFloatTensor_newWithSize3d(dst_size,dst_size,dst_size);
	interpolateTensor(Tsrc, Tdst);

	THFloatTensor *Terror = THFloatTensor_newWithSize3d(dst_size,dst_size,dst_size);
	

	THFloatTensor_cadd(Terror, Tdst, -1.0, Tref);
	THFloatTensor_abs(Tdst, Terror);
	float error = THFloatTensor_meanall(Tdst);
	cout<<"Scaling error: "<<error<<" sumall = "<<THFloatTensor_sumall(Tref)<<"\n";

	float dt =0.1, sum_0 = 0., sum_1 = 0.;
	THFloatTensor *T0 = THFloatTensor_newWithSize3d(src_size,src_size,src_size);
	fillTensor(T0, f2);
	
	THFloatTensor *Tnumgrad = THFloatTensor_newWithSize3d(src_size,src_size,src_size);
	THFloatTensor_zero(Tnumgrad);
	THFloatTensor *Tangrad = THFloatTensor_newWithSize3d(src_size,src_size,src_size);
	THFloatTensor_zero(Tangrad);
	
	THFloatTensor_fill(Tdst, 1.0);
	extrapolateTensor(Tangrad, Tdst);


	for(int x=0; x<Tsrc->size[0]; x++){
		for(int y=0; y<Tsrc->size[0]; y++){
			for(int z=0; z<Tsrc->size[0]; z++){
				interpolateTensor(T0, Tdst);
				sum_0 = THFloatTensor_sumall(Tdst);
				THFloatTensor_set3d(T0, x, y, z, THFloatTensor_get3d(T0, x,y,z) + dt);
				interpolateTensor(T0, Tdst);
				sum_1 = THFloatTensor_sumall(Tdst);
				THFloatTensor_set3d(T0, x, y, z, THFloatTensor_get3d(T0, x,y,z) - dt);
				THFloatTensor_set3d(Tnumgrad, x, y, z, (sum_1 - sum_0)/dt);
				std::cout<<"("<<x<<","<<y<<","<<z<<") = "<<THFloatTensor_get3d(Tnumgrad, x, y, z)<<" vs "<<THFloatTensor_get3d(Tangrad, x, y, z)<<"\n";
			}
		}
	}
	
	THFloatTensor *Terror_grad = THFloatTensor_newWithSize3d(src_size,src_size,src_size);
	THFloatTensor_zero(Terror_grad);

	THFloatTensor_cadd(Terror_grad, Tangrad, -1.0, Tnumgrad);
	THFloatTensor_abs(Tsrc, Terror_grad);
	float grad_error = THFloatTensor_meanall(Tsrc);
	cout<<"Scaling error: "<<grad_error<<"\n";


	return 1;
}