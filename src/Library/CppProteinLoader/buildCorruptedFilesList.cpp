#include "cProteinLoader.h"
#include <string>
#include <iostream>
#include <fstream>
#include <exception>

#include "TH/TH.h"

int main(){
	cProteinLoader pL;
	std::ifstream pfile("../protListTrainAndValidation.txt");
	std::string line;
	std::ofstream sfile("../corruptedList.dat");

	THFloatTensor *grid = THFloatTensor_newWithSize4d(4,120,120,120);

	while ( getline (pfile,line) ){
		std::cout<<line;
		int res = pL.loadPDB(line);
		pL.res=1.0;
		pL.computeBoundingBox();
		pL.projectToTensor(grid, true, true);
		if(res==1){
			std::cout<<"\t pass\n";
		}else{
			sfile<<line<<'\n';
			std::cout<<"\t corrupt\n";
		}
		
	}

	pfile.close();
	sfile.close();

	return 0;

}