#ifndef CPROTEIN_H_
#define CPROTEIN_H_
#include <TH.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <set>
#include <math.h>
#include <GlutFramework.h>
#include <cProteinLoader.h>
using namespace glutFramework;


class cAtom{

public:
	float x,y,z;
	int type, ind;
	
	cAtom(float x, float y, float z, int type):x(x),y(y),z(z),type(type){};
	~cAtom(){};

	Vector<float> getColor(){
		switch(type){
			case 0:
				return Vector<float>(0.5,0.5,0.5);
				break;
			case 1:
				return Vector<float>(0.2,0.2,0.8);
				break;
			case 2:
				return Vector<float>(1.0,0.2,0.2);
				break;
			case 3:
				return Vector<float>(1.0,1.0,1.0);
				break;
			default:
				return Vector<float>(0.0,1.0,0.0);
		};
	};
	Vector<float> getPosition(){return Vector<float>(x,y,z);};
	
};

class cBond {

public:
	cAtom *atom1, *atom2;
	float initLength; //initial lenght of the bond

	cBond(cAtom *atom1, cAtom *atom2):atom1(atom1),atom2(atom2){
		initLength = getLength();
	};
	cBond(cAtom *atom1, cAtom *atom2, float length):atom1(atom1),atom2(atom2),initLength(length){};
	~cBond(){};


	float getLength(){
		return sqrt((atom1->x-atom2->x)*(atom1->x-atom2->x) + (atom1->y-atom2->y)*(atom1->y-atom2->y) + (atom1->z-atom2->z)*(atom1->z-atom2->z));
	}

};

class cProtein: public Object {

public:
	std::vector<cAtom> atoms;
	std::vector<cBond> bonds;
	
	float bx0,by0,bz0,bx1,by1,bz1;

	cProtein(cProteinLoader &pL);
	~cProtein(){};

	float getDist2(cAtom &a1, cAtom &a2){return (a1.x-a2.x)*(a1.x-a2.x) + (a1.y-a2.y)*(a1.y-a2.y) + (a1.z-a2.z)*(a1.z-a2.z);};
	void display();
	Vector<double> getCenter();
	
	void shiftProtein(float dx, float dy, float dz){
		for(int i=0;i<atoms.size();i++){
			atoms[i].x+=dx;
			atoms[i].y+=dy;
			atoms[i].z+=dz;
		}
		bx0+=dx;bx1+=dx;
		by0+=dy;by1+=dy;
		bz0+=dz;bz1+=dz;
	}
	void computeBoundingBox();
	void centerBoundingBox();

	void constructBonds();
	void scale(Vector<double> r0, float resolution);
	
};


#endif /* CPROTEINLOADER_H_ */