#include <cMarchingCubes.h>
#include <cProteinLoader.h>
#include <GlutFramework.h>
#include <cProtein.h>
#include "TH/TH.h"
#include <iostream>
#include <math.h>
int main(int argc, char** argv)
{
 	THGenerator *gen = THGenerator_new();
 	THRandom_seed(gen);
 	float alpha = THRandom_uniform(gen,0,2.0*M_PI);
 	float beta = THRandom_uniform(gen,0,2.0*M_PI);
 	float theta = THRandom_uniform(gen,0,2.0*M_PI);
 	cMatrix33 random_rotation = cMatrix33::rotationXYZ(alpha,beta,theta);
 	std::cout<<random_rotation<<'\n';

	GlutFramework framework;
		
	cProteinLoader pL;
	int res = pL.loadPDB("/home/lupoglaz/ProteinsDataset/CASP/T0188/T0188TS001_1");
	pL.assignAtomTypes(2);
	
	int size = 120;
	pL.res=1.0;
	pL.computeBoundingBox();
	pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); //placing center of the bbox to the origin
	pL.rotateProtein(random_rotation);
	pL.shiftProtein( 0.5*cVector3(size,size,size)*pL.res ); // placing center of the protein to the center of the grid
	THFloatTensor *grid = THFloatTensor_newWithSize4d(pL.num_atom_types,size,size,size);

	float dx_max = fmax(0, grid->size[1]*pL.res/2.0 - (pL.b1[0]-pL.b0[0])/2.0)*0.5;
	float dy_max = fmax(0, grid->size[2]*pL.res/2.0 - (pL.b1[1]-pL.b0[1])/2.0)*0.5;
	float dz_max = fmax(0, grid->size[3]*pL.res/2.0 - (pL.b1[2]-pL.b0[2])/2.0)*0.5;
	float dx = THRandom_uniform(gen,-dx_max,dx_max);
 	float dy = THRandom_uniform(gen,-dy_max,dy_max);
 	float dz = THRandom_uniform(gen,-dz_max,dz_max);
 	pL.shiftProtein(cVector3(dx,dy,dz));

	pL.projectToTensor(grid);
	

	cProtein proteinVis(pL);	
	proteinVis.scale(Vector<double>(grid->size[1]*res/2.0, grid->size[2]*res/2.0, grid->size[3]*res/2.0), pL.res);
	
	Vector<double> lookAtPos(size/2,size/2,size/2);
    framework.setLookAt(size, size/2,size/2,lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);

	cVolume v(grid);
	framework.addObject(&v); 
	framework.addObject(&proteinVis);

    framework.startFramework(argc, argv);
    THGenerator_free(gen);
	return 0;
}