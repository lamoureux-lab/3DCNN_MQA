#include "cProteinLoader.h"
#include <iostream>
#include <string>
#include <THC/THC.h>
#include <math.h>

#define d 2
	__global__ void projectTensorToAtoms(float* d_flat_data, int* d_n_atoms, size_t* d_offsets, float *grad,
                                        int batch_size, int num_atom_types, int spatial_dim,
									    float res)
    /*
    Input:
        d_flat_data: coordinates in a flat array:
            flat_data: {protein1, ... proteinN}
            protein1: {atom_type1 .. atom_typeM}
            atom_type: {x1,y1,z1 .. xL,yL,zL}
        d_n_atoms: number of atoms in each atom_type 
        d_offsets: offset for coordinates for each atom_type volume
        grad: gradient to be projected on atoms
    Output: 
        d_flat_data: coordinates are rewritten for each atom to store the gradient projection
    */
    {
		size_t func_index = threadIdx.x + blockIdx.x*blockDim.x;
		float *volume = grad + func_index * spatial_dim*spatial_dim*spatial_dim;
		float *atoms_coords = d_flat_data + d_offsets[func_index];
		int n_atoms = d_n_atoms[func_index];
		for(int atom_idx = 0; atom_idx<n_atoms; atom_idx+=3){
			float 	x = atoms_coords[atom_idx],
					y = atoms_coords[atom_idx + 1],
					z = atoms_coords[atom_idx + 2];
            atoms_coords[atom_idx] = 0.0;
            atoms_coords[atom_idx+1] = 0.0;
            atoms_coords[atom_idx+2] = 0.0;
			int x_i = floor(x/res);
			int y_i = floor(y/res);
			int z_i = floor(z/res);
			for(int i=x_i-d; i<=(x_i+d);i++){
				for(int j=y_i-d; j<=(y_i+d);j++){
					for(int k=z_i-d; k<=(z_i+d);k++){
						if( (i>=0 && i<spatial_dim) && (j>=0 && j<spatial_dim) && (k>=0 && k<spatial_dim) ){
							int idx = k + j*spatial_dim + i*spatial_dim*spatial_dim;							
							float r2 = (x - i*res)*(x - i*res)+\
							(y - j*res)*(y - j*res)+\
							(z - k*res)*(z - k*res);
							atoms_coords[atom_idx] -= i*res*volume[idx]*exp(-r2/2.0);
                            atoms_coords[atom_idx + 1] -= j*res*volume[idx]*exp(-r2/2.0);
                            atoms_coords[atom_idx + 2] -= k*res*volume[idx]*exp(-r2/2.0);
						}
					}
				}
			}
		}
	}

extern "C"{
	int prepareProtein(const char* proteinPath, 
					float resolution, 
					int assigner_type, int spatial_dim, 
					float** data_pointer, int* n_atoms){
        /*
        Prepares protein for projecting on the grid.
        */
		cProteinLoader pL;
		pL.loadPDB(proteinPath);
		if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
		
		pL.res = resolution;
		pL.computeBoundingBox();
		//placing center of the bbox to the origin
		pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); 
		// placing center of the protein to the center of the grid
		pL.shiftProtein( 0.5*cVector3(spatial_dim, spatial_dim, spatial_dim)*pL.res ); 

		for(int i=0; i<pL.num_atom_types; i++){
			std::vector<float> coords; // vector of plain coords of a particular atom type
			for(int j=0; j<pL.atomType.size();j++){
				if(pL.atomType[j]==i){
					coords.push_back(pL.r[j].v[0]);
					coords.push_back(pL.r[j].v[1]);
					coords.push_back(pL.r[j].v[2]);
				}
			}
			float *coords_plain = new float[coords.size()];
			data_pointer[i] = coords_plain;
			n_atoms[i] = coords.size();
			for(int j=0; j<coords.size();j++)
				coords_plain[j]=coords[j];
		}
		return 1;
	}

    int saveProtein(const char* proteinPath, 
					int assigner_type,
					float** data_pointer, int* n_atoms){

		cProteinLoader pL;
		pL.loadPDB(proteinPath);
		if(pL.assignAtomTypes(assigner_type)<0){
			return -1;
		}
        pL.dr.resize(pL.r.size());
		std::cout<<"Assigned atom types"<<std::endl;
		for(int i=0; i<pL.num_atom_types; i++){
            std::vector<float> indexes; // vector of atom indexes for plain coords
			for(int j=0; j<pL.atomType.size(); j++){
				if(pL.atomType[j]==i){
					indexes.push_back(j);
					indexes.push_back(j);
					indexes.push_back(j);
				}
			}
            for(int j=0; j<n_atoms[i]; j+=3){
                cVector3 dr(data_pointer[i][j], data_pointer[i][j+1], data_pointer[i][j+2]);
                pL.dr[indexes[j]] = dr;
            }
		}
        pL.savePDB(proteinPath);
		return 1;
	}

	typedef struct{
		char **strings;
		size_t len;
		size_t ind;
	} batchInfo;

	batchInfo* createBatchInfo(int batch_size){
		//std::cout<<"Creating batch info of size = "<<batch_size<<"\n";
		batchInfo *binfo;
		binfo = new batchInfo;
		binfo->strings = new char*[batch_size];
		binfo->len = batch_size;
		binfo->ind = 0;
		return binfo;
	}

	void deleteBatchInfo(batchInfo* binfo){
		for(int i=0;i<binfo->len;i++){
			delete [] binfo->strings[i];
		}
		binfo->len=0;
		binfo->ind=0;
		delete binfo;
	}

	void pushProteinToBatchInfo(const char* filename, batchInfo* binfo){
		std::string str(filename);
		//std::cout<<"Pushing the string "<<str<<" to the position "<<pos<<"\n";
		//std::cout<<grid4D->nDimension<<"\n";

		binfo->strings[binfo->ind] = new char[str.length()+1];
		for(int i=0; i<str.length(); i++){
			binfo->strings[binfo->ind][i] = str[i];
		}
		binfo->strings[binfo->ind][str.length()]='\0';
		binfo->ind += 1;
		//std::cout<<binfo->grids4D[pos]->nDimension<<"\n";
	}

	void printBatchInfo(batchInfo* binfo){
		for(int i=0;i<min(binfo->len,binfo->ind);i++){
			std::cout<<binfo->strings[i]<<"\n";
		}
	}
	

	int getGradientsCUDA(THCState *state,
						 batchInfo* batch, THCudaTensor *batch5D,
						 float resolution,
						 int assigner_type, int spatial_dim){
 		int num_atom_types;
 		if(assigner_type==1)num_atom_types=4;
 		else num_atom_types=11;
		// std::cout<<"Launched function"<<(batch->len)*num_atom_types<<std::endl;
 		float **data_array = new float*[(batch->len)*num_atom_types];
 		int *n_atoms = new int[(batch->len)*num_atom_types];
 		size_t *offsets = new size_t[(batch->len)*num_atom_types];
 		
 		std::vector<int> flags;
 		flags.resize(batch->len);
		#pragma omp parallel for num_threads(10)
		for(int i=0; i<batch->len; i++){
			int res = prepareProtein(   batch->strings[i], resolution, assigner_type, spatial_dim, 
				                        data_array + i*num_atom_types, n_atoms + i*num_atom_types);
			flags[i] = res;
		}

		for(int i=0; i<batch->len; i++){
			if(flags[i]<0){
				std::cout<<"Corrupt file detected\n";
				for(int j=0;j<batch->len;j++){
					if(flags[j]>0)
						for(int k=0;k<num_atom_types;k++)
							delete[] data_array[k+j*num_atom_types];	
				}
				delete[] data_array;
				delete[] n_atoms;
				delete[] offsets;
				return -1;
			}
		}

		size_t total_size = 0;
		for(int batch_idx=0; batch_idx<batch->len; batch_idx++){
			for(int a_type_idx=0; a_type_idx<num_atom_types; a_type_idx++){
				int volume_idx = a_type_idx + batch_idx*num_atom_types;
				if(volume_idx>0)
					offsets[volume_idx] = offsets[volume_idx - 1] + n_atoms[volume_idx-1];
				else
					offsets[volume_idx] = 0;
				total_size += n_atoms[volume_idx];
			}
		}

		//data_array copy to gpu
		float *d_flat_data;
		int *d_n_atoms;
		size_t *d_offsets;
		cudaMalloc( (void**) &d_flat_data, total_size*sizeof(float) );
		cudaMalloc( (void**) &d_n_atoms, (batch->len)*num_atom_types*sizeof(int) );
		cudaMalloc( (void**) &d_offsets, (batch->len)*num_atom_types*sizeof(size_t) );
				
		cudaMemcpy( d_n_atoms, n_atoms, (batch->len)*num_atom_types*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy( d_offsets, offsets, (batch->len)*num_atom_types*sizeof(size_t), cudaMemcpyHostToDevice);
		
		for(int i=0; i<batch->len; i++){
			for(int j=0; j<num_atom_types; j++){
				int volume_idx = j+i*num_atom_types;
				cudaMemcpy( d_flat_data + offsets[volume_idx], data_array[j+i*num_atom_types], 
							n_atoms[j+i*num_atom_types]*sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		float* grid = THCudaTensor_data(state, batch5D);
		std::cout<<"Start project gradient"<<std::endl;
		projectTensorToAtoms<<<(batch->len), num_atom_types>>>(	d_flat_data, d_n_atoms, d_offsets,
															    grid, batch->len, num_atom_types, spatial_dim,
															    resolution);
		std::cout<<"Projected gradient"<<std::endl;
        for(int i=0; i<batch->len; i++){
			for(int j=0; j<num_atom_types; j++){
				int volume_idx = j+i*num_atom_types;
				cudaMemcpy( data_array[j+i*num_atom_types], d_flat_data + offsets[volume_idx],
							n_atoms[j+i*num_atom_types]*sizeof(float), cudaMemcpyDeviceToHost);
			}
		}
		std::cout<<"Copied memory back"<<std::endl;
        // #pragma omp parallel for num_threads(10)
		for(int i=0; i<batch->len; i++){
			saveProtein(batch->strings[i], assigner_type,
				        data_array + i*num_atom_types, n_atoms + i*num_atom_types);			
		}
		std::cout<<"Saved protein"<<std::endl;
		for(int i=0;i<batch->len*num_atom_types;i++){
			delete[] data_array[i];
		}
		delete[] data_array;
		delete[] n_atoms;
		delete[] offsets;
		cudaFree(d_n_atoms);
		cudaFree(d_flat_data);
		cudaFree(d_offsets);
		return 1;
	}
}