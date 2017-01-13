#include "cMarchingCubes.h"
#include <cProteinLoader.h>
#include <GlutFramework.h>
#include <cProtein.h>
#include <TensorMathUtils.h>

extern "C" void visualizeTensorAndProtein(const char* proteinPath, THFloatTensor *tensor)
{
	GlutFramework framework;
	int default_size = 120;

	cProteinLoader pL;
	int res = pL.loadPDB(proteinPath);
	pL.assignAtomTypes(2);
	
	int size = 120;
	pL.res=1.0;
	pL.computeBoundingBox();
	pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); //placing center of the bbox to the origin
	pL.shiftProtein( 0.5*cVector3(size,size,size)*pL.res ); // placing center of the protein to the center of the grid
	

	cProtein proteinVis(pL);	
	//proteinVis.scale(Vector<double>(tensor->size[1]*res/2.0, tensor->size[2]*res/2.0, tensor->size[3]*res/2.0), pL.res);
	
	Vector<double> lookAtPos(size/2,size/2,size/2);
    framework.setLookAt(size, size/2,size/2,lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);

	cVolume v(tensor);
	framework.addObject(&v); 
	framework.addObject(&proteinVis);

	char* argv[] = {"main",""};
    int argc = 0;
    framework.startFramework(argc, argv);
}