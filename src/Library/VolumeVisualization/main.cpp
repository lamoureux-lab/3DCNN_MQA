#include <cMarchingCubes.h>
#include <cProteinLoader.h>
#include <GlutFramework.h>
int main(int argc, char** argv)
{
	GlutFramework framework;
	
	cProteinLoader pL;
	THFloatTensor *grid = THFloatTensor_newWithSize4d(4,120,120,120);
	int res = pL.loadPDB("/home/lupoglaz/ProteinsDataset/RosettaDataset/1aa2/1aa2.pdb");
	pL.res=1.0;
	pL.computeBoundingBox();
	pL.projectToTensor(grid, false, false);

	std::cout<<res<<" "<<THFloatTensor_sumall(grid)<<"\n";
	
	Vector<double> lookAtPos(60,60,60);
    framework.setLookAt(120, 60.0, 60.0,lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);

	cVolume v(grid);
	framework.addObject(&v);

    framework.startFramework(argc, argv);
	return 0;
}
