
#include <iostream>
#include <fstream>
#include <string.h>

#include <TH.h>

extern "C" void saveTensorToCPP4(THFloatTensor *tensor, std::string filename){

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