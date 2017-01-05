/* 
 * File:   cBBox.h
 * Author: george
 *
 * Created on July 31, 2012, 4:27 PM
 */

#pragma once
#include "cVector3.hpp"
#include "cMatrix33.hpp"
#include "flags.h"


#include <vector>
using namespace std;


class cBBox {
public:
    cBBox();
    cBBox(const cVector3 _r0, const cVector3 _r1);
    cBBox(const cBBox& orig);
    virtual ~cBBox();
    
public:
    cVector3 r0,r1;

    double getSize() const;

    cVector3 getCenter() const {return 0.5*(r0+r1);}

    void uniteWith(const cBBox &b);

    bool inBox(const cVector3 &r) const;
    
    void addVector(const cVector3 &r);
    
    void broaden(double dr);
    
    void setZero();
    
    void boundVectors(cVector3 *vec, int N);
    void boundVectorsRotations(cVector3 *vec, int N);
    void boundVectorsRotations(vector<cVector3> &vecs){boundVectorsRotations(vecs.data(),vecs.size());};
    
    void boundRotated(cMatrix33 &rot);
    void boundAllRotations();
    double getVolume() const;
    
    cVector3 getRandomVector();
    
    bool operator ==(const cBBox& box)const;
    bool operator !=(const cBBox& box)const;
    
private:

};


