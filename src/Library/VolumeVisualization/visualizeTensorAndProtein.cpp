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
	cProtein proteinVis(pL);
	
	proteinVis.centerBoundingBox();
	proteinVis.shiftProtein(default_size/2.0,default_size/2.0,default_size/2.0);
		
    THFloatTensor *Trescaled = THFloatTensor_newWithSize3d(default_size,default_size,default_size);
    interpolateTensor(tensor, Trescaled);

	cVolume v(Trescaled);
	framework.addObject(&v);
	framework.addObject(&proteinVis);
	

	Vector<double> lookAtPos(float(default_size)*0.5, float(default_size)*0.5, float(default_size)*0.5);
    framework.setLookAt(default_size, float(default_size)*0.5, float(default_size)*0.5, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);

	char* argv[] = {"main",""};
    int argc = 0;
    framework.startFramework(argc, argv);
}