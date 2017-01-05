
#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <math.h>
#include <set>
#include "cProtein.h"

cProtein::cProtein(cProteinLoader &pL){

	//atoms
	for(int i=0;i<pL.atomType.size();i++){
		atoms.push_back(cAtom(pL.r[i][0],pL.r[i][1],pL.r[i][2],pL.atomType[i]));
	}
	constructBonds();

}

void cProtein::constructBonds(){
	//bonds

	for(int i=0;i<atoms.size();i++){
		for(int j=i+1;j<atoms.size();j++){
			if(getDist2(atoms[i],atoms[j])<4.0){
				bonds.push_back(cBond(&atoms[i],&atoms[j]));
			}
		}
	}

}


void cProtein::display(){
	glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);
        //std::cout<<"atoms"<<std::endl;
		glBegin(GL_POINTS);
		for(int i=0;i<atoms.size();i++){
			Vector<float> color(atoms[i].getColor());
			glColor3f(color.x,color.y,color.z);
			glVertex3f(atoms[i].x,atoms[i].y,atoms[i].z);
		}
		glEnd();
		//std::cout<<"bonds"<<std::endl;
		glLineWidth(2);
		glBegin(GL_LINES);
		for(int i=0;i<bonds.size();i++){
			Vector<float> color1(bonds[i].atom1->getColor());
			Vector<float> color2(bonds[i].atom2->getColor());
			glColor3f(color1.x,color1.y,color1.z);
			glVertex3f(bonds[i].atom1->x,bonds[i].atom1->y,bonds[i].atom1->z);
			glColor3f(color2.x,color2.y,color2.z);
			glVertex3f(bonds[i].atom2->x,bonds[i].atom2->y,bonds[i].atom2->z);
		}
		glEnd();
		
	glPopAttrib(); 

}

Vector<double> cProtein::getCenter(){
	Vector<double> center(0,0,0);
	for(int i=0;i<atoms.size();i++){
		center = center + Vector<double>(atoms[i].x,atoms[i].y,atoms[i].z);
	}
	return center/( (double)atoms.size());
}

void cProtein::centerBoundingBox(){
	computeBoundingBox();
	float box_center_x = (bx0+bx1)/2.0;
	float box_center_y = (by0+by1)/2.0;
	float box_center_z = (bz0+bz1)/2.0;

	shiftProtein(-box_center_x,-box_center_y,-box_center_z);
	//shiftProtein(-bx0,-by0,-bz0);
	//computeBoundingBox();
}


void cProtein::computeBoundingBox(){
	bx0=std::numeric_limits<float>::infinity(); by0=std::numeric_limits<float>::infinity(); bz0=std::numeric_limits<float>::infinity();
	bx1=-1*std::numeric_limits<float>::infinity(); by1=-1*std::numeric_limits<float>::infinity(); bz1=-1*std::numeric_limits<float>::infinity();
	for(int i=0;i<atoms.size();i++){
		if(atoms[i].x<bx0){
			bx0=atoms[i].x;
		}
		if(atoms[i].y<by0){
			by0=atoms[i].y;
		}
		if(atoms[i].z<bz0){
			bz0=atoms[i].z;
		}

		if(atoms[i].x>bx1){
			bx1=atoms[i].x;
		}
		if(atoms[i].y>by1){
			by1=atoms[i].y;
		}
		if(atoms[i].z>bz1){
			bz1=atoms[i].z;
		}
	}
}

void cProtein::scale(Vector<double> r0, float resolution){
	for(int i=0;i<atoms.size();i++){
		Vector<double> r = Vector<double>(atoms[i].x, atoms[i].y, atoms[i].z);
		Vector<double> r_new = (r-r0)/resolution + r0;
		atoms[i].x = r_new.x;
		atoms[i].y = r_new.y;
		atoms[i].z = r_new.z;
	}
}