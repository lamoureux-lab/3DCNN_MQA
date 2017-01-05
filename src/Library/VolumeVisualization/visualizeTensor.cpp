#include "cMarchingCubes.h"
#include <cProteinLoader.h>
#include <GlutFramework.h>

extern "C" void visualizeTensor(THFloatTensor *tensor, int size)
{
	GlutFramework framework;
		
	Vector<double> lookAtPos(float(size)*0.5, float(size)*0.5, float(size)*0.5);
    framework.setLookAt(size, float(size)*0.5, float(size)*0.5, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);

	cVolume v(tensor);
	framework.addObject(&v);
	
	char* argv[] = {"main",""};
    int argc = 0;
    framework.startFramework(argc, argv);
}
