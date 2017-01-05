/*************************************************************************\

  Copyright 2010 Sergei Grudinin
  All Rights Reserved.

  The author may be contacted via:

  Mail:					Sergei Grudinin
						INRIA Rhone-Alpes Research Unit
						Zirst - 655 avenue de l'Europe - Montbonnot
						38334 Saint Ismier Cedex - France

  Phone:				+33 4 76 61 53 24

  EMail:				sergei.grudinin@inria.fr

\**************************************************************************/

#include <iostream>
#include <fstream>

#include <string.h>
#include "cDensityMap.hpp"

double cDensityMap::getDensity(int i, int j, int k)const {

	return density[i][j][k];

}

cDensityMap::cDensityMap() {

	density = NULL;
	densityVector = NULL;
	inside = NULL;

}

cDensityMap::~cDensityMap() {};


cBBox cDensityMap::getCropBBox(double cropThreshold){
    cBBox cropBox;
    bool accumulate=false;
    for (int i=0;i<grid.M[0];i++){
	    for (int j=0;j<grid.M[1];j++){
	        for (int k=0;k<grid.M[2];k++){
                if(density[i][j][k]>cropThreshold && accumulate){
                    cropBox.addVector(grid.indexToVector(i,j,k));
                }
                if(density[i][j][k]>cropThreshold && !accumulate){
                    cropBox.r0 = grid.indexToVector(i,j,k);
                    cropBox.r1 = cropBox.r0;
                    accumulate=true;
                }
            }}}
    return cropBox;
}

void cDensityMap::computeAvMaxMinDensity()
{
    avDensity=0;
	minDensity=0.0;
	maxDensity=0.0;
	for (int i=0;i<grid.M[0];i++){
	    for (int j=0;j<grid.M[1];j++){
	        for (int k=0;k<grid.M[2];k++){
	            avDensity+=density[i][j][k];
	            if(density[i][j][k]<minDensity)
                    minDensity=density[i][j][k];
                if(density[i][j][k]>maxDensity)
                    maxDensity=density[i][j][k];
	        }
	    }
	}
	avDensity/=grid.getTotalSize();
}


void cDensityMap::regularizeEMDensity()
{
    double sigma=0.0;
    int Ntotal=0;
    for (int i=0;i<grid.M[0];i++){
	    for (int j=0;j<grid.M[1];j++){
	        for (int k=0;k<grid.M[2];k++){
	            Ntotal+=1;
	            sigma+=(density[i][j][k]-avDensity)*(density[i][j][k]-avDensity);
	        }}}
    sigma=sqrt(sigma)/((double)Ntotal);
    for (int i=0;i<grid.M[0];i++){
	    for (int j=0;j<grid.M[1];j++){
	        for (int k=0;k<grid.M[2];k++){
                if(density[i][j][k]<avDensity+2.0*sigma)
                    density[i][j][k]=0.0;
	        }}}
}

cVector3 cDensityMap::getWeightedCenter()
{
    cVector3 center(0,0,0);
    double M=0.0;
    for (int i=0;i<grid.M[0];i++){
	    for (int j=0;j<grid.M[1];j++){
	        for (int k=0;k<grid.M[2];k++){
                cVector3 r;    
	        mapToWorld(i,j,k,r);
	        center+=r*density[i][j][k];
                M+=density[i][j][k];
	        }
	    }
    }
    return center/M;
}

void cDensityMap::worldToMap(const cVector3 &world, int &i, int &j, int &k)const
{

	double zSize = (world.v[2]-origin.v[2]) / axes[2].v[2];
	k=(int)floor(zSize / voxel.sizeVector.v[2]);

	double ySize = ( world.v[1] - origin.v[1] - zSize*axes[2].v[1]) / axes[1].v[1];
	j=(int)floor(ySize / voxel.sizeVector.v[1]);

	double xSize = ( world.v[0] - origin.v[0] - zSize*axes[2].v[0] - ySize*axes[1].v[0]) / axes[0].v[0];
	i=(int)floor(xSize / voxel.sizeVector.v[0]);
    
    

}

void cDensityMap::mapToWorld(int &i, int &j, int &k, cVector3 &world)const {
	world = origin + voxel.sizeVector.v[0]* (i)* axes[0] +
                    voxel.sizeVector.v[1]*(j)* axes[1] +
                    voxel.sizeVector.v[2]*(k)* axes[2];
    
}
double cDensityMap::getAverageValue(const cVector3& r) const{
    int i,j,k;
	cVector3 voxelOrigin;
	double densityNeighbor[2][2][2];

	worldToMap(r,i,j,k);
	if (i<0 || j <0 || k<0)
		return 0.0;
	if ((i>grid.M[0]-2) || (j>grid.M[1]-2) || (k>grid.M[2]-2))
		return 0.0;

	mapToWorld(i,j,k,voxelOrigin);

	double zSize = (r.v[2]-voxelOrigin.v[2]) / axes[2].v[2];
	double z= (zSize / voxel.sizeVector.v[2]);

	double ySize = ( r.v[1] - voxelOrigin.v[1] - zSize*axes[2].v[1]) / axes[1].v[1];
	double y= (ySize / voxel.sizeVector.v[1]);

	double xSize = ( r.v[0] - voxelOrigin.v[0] - zSize*axes[2].v[0] - ySize*axes[1].v[0]) / axes[0].v[0];
	double x= (xSize / voxel.sizeVector.v[0]);

	//double x= ((r-voxelOrigin) | axes[0]) /voxel.sizeVector.v[0];
	//double y=((r-voxelOrigin) | axes[1])/voxel.sizeVector.v[1];
	//double z=((r-voxelOrigin) | axes[2])/voxel.sizeVector.v[2];

	densityNeighbor[0][0][0]=density[i+0][j+0][k+0];
	densityNeighbor[0][0][1]=density[i+0][j+0][k+1];

	densityNeighbor[0][1][0]=density[i+0][j+1][k+0];
	densityNeighbor[0][1][1]=density[i+0][j+1][k+1];

	densityNeighbor[1][0][0]=density[i+1][j+0][k+0];
	densityNeighbor[1][0][1]=density[i+1][j+0][k+1];

	densityNeighbor[1][1][0]=density[i+1][j+1][k+0];
	densityNeighbor[1][1][1]=density[i+1][j+1][k+1];

	double m1mx=1-x;
	double m1my=1-y;
	double m1mz=1-z;

	//return densityNeighbor[0][0][0];

	return densityNeighbor[0][0][0]*m1mx*m1my*m1mz+
		   densityNeighbor[1][0][0]*x*m1my*m1mz+
		   densityNeighbor[0][1][0]*m1mx*y*m1mz+
		   densityNeighbor[0][0][1]*m1mx*m1my*z+
		   densityNeighbor[1][0][1]*x*m1my*z+
		   densityNeighbor[0][1][1]*m1mx*y*z+
		   densityNeighbor[1][1][0]*x*y*m1mz+
		   densityNeighbor[1][1][1]*x*y*z;

}
double cDensityMap::getValue(const cVector3 &r)const {
    
    if(!grid.bbox.inBox(r))
        return 0.0;
    
    cVector3 idx = grid.vectorToIndex(r);
//    cout<<(r-grid.bbox.r0)/grid.dh<<"\n";
//    cout<<idx<<"\n";
//    if(!grid.isIndexOnGrid(idx))
//        return 0.0;
    
    return getDensity(idx.v[0],idx.v[1],idx.v[2]);
}
double cDensityMap::getValue(double x, double y, double z) const{
    return this->getValue(cVector3(x,y,z));
}


void cDensityMap::updateBin() {

	double diff = maxDensity - minDensity;
	double delta = diff/(127+1);

	isoBin = (isoValue - minDensity) / delta;
	if (isoBin > 0)
		isoBin --;
}

void cDensityMap::meanDensity(float kT) {
	float beta = 1./kT;
	for (unsigned int i=0;i<grid.M[0];i++)
		for (unsigned int j=0;j<grid.M[1];j++)
			for (unsigned int k=0;k<grid.M[2];k++) {
				double val = 0;
				for (int type = 0; type < nTypes; type++) {
					if (-beta*densityVector[i][j][k][type] < 100)
						val += exp(-beta*densityVector[i][j][k][type]);
				}
				density[i][j][k] = -kT*log(val);
			}

}

void cDensityMap::normalize(int nChains) {
	for (unsigned int i=0;i<grid.M[0];i++)
		for (unsigned int j=0;j<grid.M[1];j++)
			for (unsigned int k=0;k<grid.M[2];k++)
				for (int type = 0; type < nTypes; type++)  {
					float f = nChains - inside[i][j][k];
					if (f > 0)
						densityVector[i][j][k][type] /= f;
				}

}



#define CCP4HDSIZE 1024

// density maps
void nextLine(std::ifstream *fs) {
  fs->ignore(10240, '\n');    // go on the next line
}


/* Only works with aligned 4-byte quantities, will cause a bus error */
/* on some platforms if used on unaligned data.                      */
static void swap4_aligned(void *v, long ndata) {
  int *data=(int *) v;
  long i;
  int *N;
  for (i=0; i<ndata; i++) {
    N=data + i;
    *N=(((*N>>24)&0xff) | ((*N&0xff)<<24) |
        ((*N>>8)&0xff00) | ((*N&0xff00)<<8));
  }
}

// CCP4 electron density maps reader
void cDensityMap::load(const std::string &densityMapFileName) {

	std::ifstream *densityMapFile=new std::ifstream(densityMapFileName.c_str(), std::ifstream::binary);

	//char 	tmpChar[80];

	if (densityMapFile->fail()) {
        
        throw(std::string("Error: could not open CCP4 density map file "+densityMapFileName));
		std::cerr << "Error: could not open CCP4 density map file " << densityMapFileName.c_str() << std::endl;
		return;

	}
	else std::cout << "Loading CCP4 density map from file " << densityMapFileName.c_str() << ".\n";

	int extent[3], mode, ccp4Origin[3], grid[3], crs2xyz[3];
	float cellDimensions[3], cellAngles[3];
	int symBytes;
	std::string mapString;
	char symData[80];
	unsigned int i, j;
	int xIndex, yIndex, zIndex;
	float *rawData;

	densityMapFile->read((char*)extent,3*sizeof(int));
	densityMapFile->read((char*)&mode,sizeof(int));
	densityMapFile->read((char*)ccp4Origin,3*sizeof(int));
	densityMapFile->read((char*)grid,3*sizeof(int));
	densityMapFile->read((char*)cellDimensions,3*sizeof(float));
	densityMapFile->read((char*)cellAngles,3*sizeof(float));
	densityMapFile->read((char*)crs2xyz,3*sizeof(int));

	densityMapFile->read((char*)&minDensity,sizeof(float));
	densityMapFile->read((char*)&maxDensity,sizeof(float));
	densityMapFile->read((char*)&avDensity,sizeof(float));

	// Check the number of bytes used for storing symmetry operators
	densityMapFile->seekg(92, std::ios::beg);
	densityMapFile->read((char*)&symBytes,sizeof(int));

	// Check for the string "MAP" at byte 208, indicating a CCP4 file.
	densityMapFile->seekg(208, std::ios::beg);
	//mapString.reserve(3*sizeof(char));
	(*densityMapFile) >> mapString;
	//std::cout<<mapString<<"\n";
	if (mapString.find("MAP")==mapString.npos)
		std::cerr << "Error: 'MAP' string missing, not a valid CCP4 file " << std::endl;

	int swap=0;
	// Check the data type of the file.
	if (mode != 2) {
		// Check if the byte-order is flipped
		swap4_aligned(&mode, 1);
		if (mode != 2) {
			std::cerr << "Error: Non-real (32-bit float) data types are unsupported " << std::endl;
			return;
		} else {
			swap=1; // enable byte swapping
		}
	}

	// Swap all the information obtained from the header
	if (swap==1) {
		swap4_aligned(extent, 3);
		swap4_aligned(ccp4Origin, 3);
		swap4_aligned(grid, 3);
		swap4_aligned(cellDimensions, 3);
		swap4_aligned(cellAngles, 3);
		swap4_aligned(crs2xyz, 3);
		swap4_aligned(&symBytes, 1);
	}

	std::cout<<"extent = ["<<extent[0]<<" x "<<extent[1]<<" x "<<extent[2]<<"]\n";
	//std::cout<<"mode="<<mode<<" \n";
	std::cout<<"ccp4Origin = ["<<ccp4Origin[0]<<" x "<<ccp4Origin[1]<<" x "<<ccp4Origin[2]<<"]\n";
	std::cout<<"grid size = ["<<grid[0]<<" x "<<grid[1]<<" x "<<grid[2]<<"]\n";
	std::cout<<"cellDimensions = ["<<cellDimensions[0]<<" x "<<cellDimensions[1]<<" x "<<cellDimensions[2]<<"]\n";
	std::cout<<"cellAngles = ["<<cellAngles[0]<<" x "<<cellAngles[1]<<" x "<<cellAngles[2]<<"]\n";
	std::cout<<"crs2xyz = ["<<crs2xyz[0]<<" x "<<crs2xyz[1]<<" x "<<crs2xyz[2]<<"]\n";
	//std::cout<<"symBytes="<<symBytes<<" \n";

	std::cout<<"minDensity = "<<minDensity<<",";
	std::cout<<"maxDensity = "<<maxDensity<<",";
	std::cout<<"avDensity = "<<avDensity<<" \n";

	// Check the dataOffset: this fixes the problem caused by files claiming
	// to have symmetry records when they do not.
	int filesize, dataOffset;
	densityMapFile->seekg(0, std::ios::end);
	filesize=densityMapFile->tellg();
	dataOffset=filesize - 4*(extent[0]*extent[1]*extent[2]);
	if (dataOffset != (CCP4HDSIZE + symBytes)) {
		if (dataOffset==CCP4HDSIZE) {
			// Bogus symmetry record information
			std::cout << "Warning: file contains bogus symmetry record. "  << std::endl;
			symBytes=0;
		} else if (dataOffset < CCP4HDSIZE) {
			std::cerr << "Error: File appears truncated and doesn't match header. "  << std::endl;
			std::cerr << "filesize = "<<filesize<<" dataOffset = "<<dataOffset<<" CCP4HDSIZE = "<<CCP4HDSIZE<<"\n";
			return ;
		} else if ((dataOffset > CCP4HDSIZE) && (dataOffset < (1024*1024))) {
			// Fix for loading SPIDER files which are larger than usual
			// In this specific case, we must absolutely trust the symBytes record
			dataOffset=CCP4HDSIZE + symBytes;
			std::cout << "Warning: File is larger than expected and doesn't match header. "  << std::endl;
			std::cout << "Warning: Continuing file load, good luck! "  << std::endl;
		} else {
			std::cerr << "Error: File is MUCH larger than expected and doesn't match header. "  << std::endl;
			return;
		}
	}

	// Read symmetry records -- organized as 80-byte lines of text.
	if (symBytes != 0) {
		std::cout << "Symmetry records found: "  << std::endl;
		densityMapFile->seekg(CCP4HDSIZE, std::ios::beg);
		for (i=0; i < symBytes/80; i++) {
			densityMapFile->read(symData,80);
			symData[79]=NULL;
			std::cout << " \t "<< symData << std::endl;
		}
	}

	// check extent and grid interval counts
	if (grid[0]==0 && extent[0] > 0) {
		grid[0]=extent[0] - 1;
		std::cout << "Warning: Fixed X interval count"  << std::endl;
	}
	if (grid[1]==0 && extent[1] > 0) {
		grid[1]=extent[1] - 1;
		std::cout << "Warning: Fixed Y interval count"  << std::endl;
	}
	if (grid[2]==0 && extent[2] > 0) {
		grid[2]=extent[2] - 1;
		std::cout << "Warning: Fixed Z interval count"  << std::endl;
	}

	// Mapping
	if (crs2xyz[0]==0 && crs2xyz[1]==0 && crs2xyz[2]==0) {
		std::cout << "  Warning: All crs2xyz records are zero."  << std::endl;
		std::cout << "  Warning: Setting crs2xyz to 1, 2, 3"  << std::endl;
		crs2xyz[0]=1;
		crs2xyz[1]=2;
		crs2xyz[2]=3;
	}

	int index[3];
	index[crs2xyz[0]-1]=0;
	index[crs2xyz[1]-1]=1;
	index[crs2xyz[2]-1]=2;

	xIndex=index[0];
	yIndex=index[1];
	zIndex=index[2];


	if (cellDimensions[0]==0.0 &&
		cellDimensions[1]==0.0 &&
		cellDimensions[2]==0.0) {
			std::cout << " Warning: Cell dimensions are all zero."  << std::endl;
			std::cout << "Setting to 1.0, 1.0, 1.0 for viewing."  << std::endl;
			std::cout << "Warning: Map file will not align with other structures."  << std::endl;
			cellDimensions[0]=1.0;
			cellDimensions[1]=1.0;
			cellDimensions[2]=1.0;
		}

	voxel.sizeVector.v[0]=cellDimensions[0] / (float) (grid[0]);
	voxel.sizeVector.v[1]=cellDimensions[1] / (float) (grid[1]);
	voxel.sizeVector.v[2]=cellDimensions[2] / (float) (grid[2]);


	//this->grid.M[0]=extent[xIndex];
	//this->grid.M[1]=extent[yIndex];
	//this->grid.M[2]=extent[zIndex];
    

	std::cout << "voxel size = ";
	voxel.sizeVector.print();

	densityMapFile->seekg(dataOffset, std::ios::beg);
	rawData=new float[extent[0]];

	density=new double**[extent[xIndex]];
	for (i=0;i<extent[xIndex];i++) {

		density[i]=new double*[extent[yIndex]];
		for (j=0;j<extent[yIndex];j++) density[i][j]=new double[extent[zIndex]];

	}

	for (index[2]=0; index[2] <extent[2]; index[2]++) {
		for (index[1]=0; index[1] <extent[1]; index[1]++) {
			// Read an entire row of data from the file, then write it into the
			// datablock with the correct slice ordering.

			densityMapFile->read((char*)rawData,sizeof(float)*extent[0]);

			for (index[0]=0; index[0] <extent[0]; index[0]++) {
				//std::cout << "index " <<index[xIndex]<<" "<<index[yIndex]<<" "<<index[zIndex]<< std::endl;
				density[index[xIndex]][index[yIndex]][index[zIndex]]=rawData[index[0]];
			}
		}
	}

	if (swap==1)
		swap4_aligned(density, extent[xIndex]*extent[yIndex]*extent[zIndex]);

	delete [] rawData;


	delete densityMapFile;

	double conv = atan(1.0)*4.0/180.0;
	double alph = cellAngles[0]*conv;
	double bet = cellAngles[1]*conv;
	double gamm = cellAngles[2]*conv;
	double sina = sin(alph);
	double cosa = cos(alph);
	double sinb = sin(bet);
	double cosb = cos(bet);
	double sing = sin(gamm);
	double cosg = cos(gamm);
	double   cosas = (cosg*cosb-cosa)/ (sinb*sing);
	double sinas = sqrt(1.0-cosas*cosas);
	double cosbs = (cosa*cosg-cosb)/ (sina*sing);
	double sinbs = sqrt(1.0-cosbs*cosbs);
	double cosgs = (cosa*cosb-cosg)/ (sina*sinb);
	double sings = sqrt(1.0-cosgs*cosgs);
	double cos2 = (cosa - cosb*cosg)/ (sing);
	double cos3 = sqrt(1-cosb*cosb - cos2*cos2);


	/*
	axes[0].v[0] = 1.0;
	axes[0].v[1] = cellDimensions[1]/cellDimensions[0]*cosg;
	axes[0].v[2] = cellDimensions[2]/cellDimensions[0]*cosb;

	axes[1].v[0] = 0.0;
	axes[1].v[1] = 1.0*sing;
	axes[1].v[2] =  -cellDimensions[2]/cellDimensions[1]*sinb*cosas;

	axes[2].v[0] = 0.0;
	axes[2].v[1] = 0.0;
	axes[2].v[2] = 1.0*sinb*sinas;
	*/

	axes[0].v[0] = 1;
	axes[0].v[1] = 0;
	axes[0].v[2] = 0;

	axes[1].v[0] =  cosg;
	axes[1].v[1] =  sing;
	axes[1].v[2] =  0;

	axes[2].v[0] = cosb;
	axes[2].v[1] = cos2;
	axes[2].v[2] = cos3;

	origin.v[0]= axes[0].v[0] * voxel.sizeVector.v[0]*(float)  ccp4Origin[xIndex]
		+  axes[1].v[0] * voxel.sizeVector.v[1]*(float)  ccp4Origin[yIndex]
		+ axes[2].v[0] * voxel.sizeVector.v[2]*(float)  ccp4Origin[zIndex];
	origin.v[1]=
		+  axes[1].v[1] * voxel.sizeVector.v[1]*(float)  ccp4Origin[yIndex]
		+ axes[2].v[1] * voxel.sizeVector.v[2]*(float)  ccp4Origin[zIndex];
	origin.v[2]=
		 axes[2].v[2] * voxel.sizeVector.v[2]*(float)  ccp4Origin[zIndex];
    for(int i=0;i<3;origin.v[i]+=0.5*voxel.sizeVector.v[i],i++);

    /// INSERTED BY GEORGE >>>>>>>

    cVector3 corner;
    corner.v[0]=axes[0].v[0] * voxel.sizeVector.v[0]*(float)  (extent[xIndex])
		+  axes[1].v[0] * voxel.sizeVector.v[1]*(float)  (extent[yIndex])
		+ axes[2].v[0] * voxel.sizeVector.v[2]*(float)  (extent[zIndex]);
    corner.v[1]=
		+  axes[1].v[1] * voxel.sizeVector.v[1]*(float)  (extent[yIndex])
		+ axes[2].v[1] * voxel.sizeVector.v[2]*(float)  (extent[zIndex]);
    corner.v[2]=
		 axes[2].v[2] * voxel.sizeVector.v[2]*(float)  (extent[zIndex]);
    for(int i=0;i<3;corner.v[i]+=0.5*voxel.sizeVector.v[i],i++);

    this->grid.bbox.r0=origin;
    this->grid.bbox.r1=corner+origin;
    this->grid.setSize(extent[xIndex],extent[yIndex],extent[zIndex]);

    //std::cout << "diagonal = " ;
    //corner.print();

    /// <<<<<INSERTED BY GEORGE

	std::cout << "origin = ";
	origin.print();

	return;
}

// CCP4 electron density maps reader
void cDensityMap::save(const std::string &densityMapFileName) {

	std::ofstream *densityMapFile=new std::ofstream(densityMapFileName.c_str(), std::ofstream::binary);

	int extent[3], mode, ccp4Origin[3], grid[3], crs2xyz[3],index[3];

	float cellDimensions[3], cellAngles[3];
	int symBytes, ispg;
	std::string mapString;
	char symData[100];
	unsigned int i, j;
	int lwordSize=4;
/*
1      NC              # of Columns    (fastest changing in map)
2      NR              # of Rows
3      NS              # of Sections   (slowest changing in map)
*/
    for(int i=0;i<3;extent[i]=this->grid.M[i],i++);
	densityMapFile->write((char*)extent,3*lwordSize);
/*
4      MODE            Data type
*/
    mode=2;
	densityMapFile->write((char*)&mode,lwordSize);
/*
5      NCSTART         Number of first COLUMN  in map
6      NRSTART         Number of first ROW     in map
7      NSSTART         Number of first SECTION in map
*/
    /*
    int originInt[3];
    worldToMap(origin,originInt[0],originInt[1],originInt[2]);
    for(int i=0;i<3;ccp4Origin[i]=originInt[i],i++);
	densityMapFile->write((char*)ccp4Origin,3*lwordSize);
	*/
    for(int i=0;i<3;ccp4Origin[i]=origin.v[i]/voxel.sizeVector.v[i],i++);

	densityMapFile->write((char*)ccp4Origin,3*lwordSize);
/*
8      NX              Number of intervals along X
9      NY              Number of intervals along Y
10      NZ              Number of intervals along Z
*/
    for(int i=0;i<3;grid[i]=this->grid.M[i],i++);
	densityMapFile->write((char*)grid,3*lwordSize);

/*
11      X length        Cell Dimensions (Angstroms)
12      Y length                     "
13      Z length                     "
*/
    for(int i=0;i<3;cellDimensions[i]=voxel.sizeVector.v[i]*this->grid.M[i],i++);
	densityMapFile->write((char*)cellDimensions,3*lwordSize);
/*
14      Alpha           Cell Angles     (Degrees)
15      Beta                         "
16      Gamma                        "
*/
    for(int i=0;i<3;cellAngles[i]=90.0,i++);
	densityMapFile->write((char*)cellAngles,3*lwordSize);
/*
17      MAPC            Which axis corresponds to Cols.  (1,2,3 for X,Y,Z)
18      MAPR            Which axis corresponds to Rows   (1,2,3 for X,Y,Z)
19      MAPS            Which axis corresponds to Sects. (1,2,3 for X,Y,Z)
*/
    crs2xyz[0]=1;
    crs2xyz[1]=2;
    crs2xyz[2]=3;
	densityMapFile->write((char*)crs2xyz,3*lwordSize);
/*
20      AMIN            Minimum density value
21      AMAX            Maximum density value
22      AMEAN           Mean    density value    (Average)
*/
	densityMapFile->write((char*)&minDensity,lwordSize);
	densityMapFile->write((char*)&maxDensity,lwordSize);
	densityMapFile->write((char*)&avDensity,lwordSize);
/*
23      ISPG            Space group number
*/
    ispg=1;
    densityMapFile->write((char*)&ispg,lwordSize);
	// Setting to null bytes used for storing symmetry operators
/*
24      NSYMBT          Number of bytes used for storing symmetry operators
*/
	symBytes=0;
	densityMapFile->write((char*)&symBytes,lwordSize);
/*
25      LSKFLG          Flag for skew transformation, =0 none, =1 if foll
26-34   SKWMAT          Skew matrix S (in order S11, S12, S13, S21 etc) if
                        LSKFLG .ne. 0.
35-37   SKWTRN          Skew translation t if LSKFLG .ne. 0.
                        Skew transformation is from standard orthogonal
                        coordinate frame (as used for atoms) to orthogonal
                        map frame, as

                                Xo(map) = S * (Xo(atoms) - t)

38      future use       (some of these are used by the MSUBSX routines
 .          "              in MAPBRICK, MAPCONT and FRODO)
 .          "   (all set to zero by default)
 .          "
52          "
*/
    int temp=0;
    densityMapFile->write((char*)&temp,lwordSize);
    densityMapFile->write((char*)symData,lwordSize*26);
/*
53	MAP	        Character string 'MAP ' to identify file type
*/
	densityMapFile->write((char*)&temp,lwordSize);
/*
54	MACHST		Machine stamp indicating the machine type
			which wrote file
55      ARMS            Rms deviation of map from mean density
56      NLABL           Number of labels being used
..
256
*/
    int temp2[202];
    densityMapFile->write((char*)temp2,lwordSize*202);

    densityMapFile->seekp(92, std::ios::beg);
	densityMapFile->write((char*)&symBytes,lwordSize);

	// Rewrite the string "MAP" at byte 208, indicating a CCP4 file.
	mapString="MAP ";
	densityMapFile->seekp(208, std::ios::beg);
	(*densityMapFile) << mapString;

    densityMapFile->seekp(1024, std::ios::beg);

    float *rawData=new float[extent[0]];

    int cIndex,rIndex,sIndex;
	for (sIndex=0; sIndex <extent[2]; sIndex++) {
		for (rIndex=0; rIndex <extent[1]; rIndex++) {

		    for (cIndex=0; cIndex <extent[0]; cIndex++) {
				//std::cout << "index " <<index[xIndex]<<" "<<index[yIndex]<<" "<<index[zIndex]<< std::endl;
				rawData[cIndex]=density[cIndex][rIndex][sIndex];
			}

		    densityMapFile->write((char*)rawData,lwordSize*extent[0]);
		}
	}
    delete rawData;
    densityMapFile->close();
	delete densityMapFile;
	return;
}
