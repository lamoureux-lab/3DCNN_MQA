/* 
 * File:   cBBox.cpp
 * Author: george
 * 
 * Created on July 31, 2012, 4:27 PM
 */

#include "cBBox.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

cBBox::cBBox() {
    r0=cVector3(-1,-1,-1);r1=cVector3(1,1,1);
    
    srand ( time(NULL) );
}
cBBox::cBBox(const cBBox& orig) {
    r0=orig.r0;r1=orig.r1;
  
}
cBBox::cBBox(const cVector3 _r0, const cVector3 _r1){
    r0=_r0;r1=_r1;
   
}
cBBox::~cBBox() {

}

double cBBox::getSize() const{
    double dx=(r1.v[0]-r0.v[0]);
    double dy=(r1.v[1]-r0.v[1]);
    double dz=(r1.v[2]-r0.v[2]);
    double size=fmax(dx,fmax(dy,dz));
    return size;
}

void cBBox::uniteWith(const cBBox &b){
    if(b.r0.v[0]<r0.v[0])r0.v[0]=b.r0.v[0];
    if(b.r0.v[1]<r0.v[1])r0.v[1]=b.r0.v[1];
    if(b.r0.v[2]<r0.v[2])r0.v[2]=b.r0.v[2];

    if(b.r1.v[0]>r1.v[0])r1.v[0]=b.r1.v[0];
    if(b.r1.v[1]>r1.v[1])r1.v[1]=b.r1.v[1];
    if(b.r1.v[2]>r1.v[2])r1.v[2]=b.r1.v[2];
}
void cBBox::addVector(const cVector3 &r){
    if(r.v[0]<r0.v[0])r0.v[0]=r.v[0];
    if(r.v[1]<r0.v[1])r0.v[1]=r.v[1];
    if(r.v[2]<r0.v[2])r0.v[2]=r.v[2];

    if(r.v[0]>r1.v[0])r1.v[0]=r.v[0];
    if(r.v[1]>r1.v[1])r1.v[1]=r.v[1];
    if(r.v[2]>r1.v[2])r1.v[2]=r.v[2];
}

bool cBBox::inBox(const cVector3 &r) const{
    if(r0.v[0]<r.v[0] && r.v[0]<r1.v[0])
        if(r0.v[1]<r.v[1] && r.v[1]<r1.v[1])
            if(r0.v[2]<r.v[2] && r.v[2]<r1.v[2])
                return true;
    return false;
}
void cBBox::broaden(double dr){
    cVector3 drV(dr,dr,dr);
    r0-=drV;
    r1+=drV;
}
void cBBox::setZero(){
    r0=cVector3(0,0,0);
    r1=r0;
}

void cBBox::boundVectors(cVector3* vec, int N){
    setZero();
    for(int i=0;i<N;i++)addVector(vec[i]);
}

void cBBox::boundVectorsRotations(cVector3* vec, int N){
    double maxDist=0.0;
    for(int i=0;i<N;i++){
        if(vec[i].norm()>maxDist){
            maxDist=vec[i].norm();
        }
    }
    r0=cVector3(-maxDist,-maxDist,-maxDist);
    r1=-r0;
}

void cBBox::boundRotated(cMatrix33& rot){
    cVector3 d=r1-r0;
    cVector3 vertexes[8]={
        r0,
        r0+cVector3(d.v[0],0,0),
        r0+cVector3(d.v[0],d.v[1],0),
        r0+cVector3(0,d.v[1],0),
        r0+cVector3(0,0,d.v[2]),
        r0+cVector3(0,d.v[1],d.v[2]),
        r0+cVector3(d.v[0],0,d.v[2]),
        r1
    };
    setZero();
    for(int i=0;i<8;i++){
        cVector3 newVertex=rot*vertexes[i];
        addVector(newVertex);
    }
    
}
void cBBox::boundAllRotations(){
    cVector3 center = 0.5*(r0+r1);
    double d = (r1-center).norm();
    r1 = center+cVector3(d,d,d);
    r0 = center-cVector3(d,d,d);
}
cVector3 cBBox::getRandomVector(){
    cVector3 x;
    x.v[0]=(r1.v[0]-r0.v[0])*rand()/RAND_MAX + r0.v[0];
    x.v[1]=(r1.v[1]-r0.v[1])*rand()/RAND_MAX + r0.v[1];
    x.v[2]=(r1.v[2]-r0.v[2])*rand()/RAND_MAX + r0.v[2];
    return x;
}
double cBBox::getVolume() const{
    return (r1[0]-r0[0])*(r1[1]-r0[1])*(r1[2]-r0[2]);
}
bool cBBox::operator ==(const cBBox& box)const{
    if(box.r0!=r0 || box.r1!=r1)
        return false;
    return true;
}
bool cBBox::operator !=(const cBBox& box)const{
    return !((*this)==box);
}