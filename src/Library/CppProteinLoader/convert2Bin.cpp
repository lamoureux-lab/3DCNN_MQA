#include "cProteinLoader.h"
#include <string>
#include <iostream>
#include <fstream>
#include <exception>

#include "TH/TH.h"

int main(int argc, char *argv[]){
	if(argc<3){
		std::cout<<"Wrong arguments\n";
		return -1;
	}else{
		std::cout<<"Converting "<<argv[1]<<" 2 "<<argv[2]<<"\n";
	}
	std::string infile(argv[1]);
	std::string outfile(argv[2]);

	cProteinLoader pL;
	pL.loadPDB(infile);
	pL.assignAtomTypes(2);
	pL.save_binary(outfile);
	

	return 0;
}