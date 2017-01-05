/*************************************************************************\

  Copyright 2010 Sergei Grudinin
  All Rights Reserved.

  The author may be contacted via:

  Mail:					Sergei Grudinin
						INRIA Rhone-Alpes Research Unit
						Zirst - 655 avenue de l'Europe - Montbonnot
						38334 Saint Ismier Cedex - France

  Phone:				+33 4 76 61 53 24

  EMail:				sergei.grudinin@inria.fr

\**************************************************************************/


#pragma once


#include <string>
#include <set>
#include <vector>
#include <list>

#include "cVector3.hpp"
#include "cBBox.h"
#include "cDecomposition.h"

class sVoxel {

	public:
		cVector3 	sizeVector;

		const cVector3 &getSize() const {return sizeVector;};
};

class cDensityMap{

	
	public:
		typedef				float****		dataType;
		void 				init() {;};

	protected:
		typedef				short int***   insideType;
		
		int					ok;
		double				***density;
		dataType			densityVector;
		insideType			inside;
		int					nTypes;

		sVoxel				voxel;
		cVector3			origin;
		cVector3			axes[3];

		float				histogram[128];

		

public:
	void load(const std::string &densityMapFileName);
	void save(const std::string &densityMapFileName);

		cDensityMapInterface();
		virtual ~cDensityMapInterface();
        void	mapToWorld(int &i, int &j, int &k, cVector3 &world)const;
		void	worldToMap(const cVector3 &world, int &i, int &j, int &k)const;

		void				meanDensity(float kT);
		void				normalize(int nChains);
		const sizeType		&getSize() const {return grid.M;};
		const cVector3		&getOrigin() const {return origin;};

        void setOrigin(cVector3 newOrigin){origin=newOrigin;};

        void centerMap(){
            cVector3 center=grid.bbox.getCenter();
            setOrigin(getOrigin()-center);
            grid.bbox.r0-=center;
            grid.bbox.r1-=center;
        }

        //computes average, max, in densities given voxel map
        void computeAvMaxMinDensity();

        cBBox getBBox() const{return grid.bbox;};
        unsigned int getSizeX()const{return grid.M[0];};
        unsigned int getSizeY()const{return grid.M[1];};
        unsigned int getSizeZ()const{return grid.M[2];};

        //returns grid settings of the map
        //gridSettings getGS();

        //getting weighted center
        cVector3 getWeightedCenter();
        
        cBBox getCropBBox(double cropThreshold);

		sVoxel				&getVoxel() {return voxel;};
		insideType			&getInside() {return inside;};
		const int			getNTypes() const {return nTypes;};

		const cVector3		*getAxes() const {return axes;};

		dataType			&getDensityVector()  {return densityVector;}
		double***			&getDensity()  {return density;}


		double				getDensity(int i, int j, int k)const;
		//double				getDensity(const cVector3 &r)const;
        
        virtual double getValue(const cVector3& r) const;
        virtual double getValue(double x, double y, double z) const;  
        virtual double getAverageValue(const cVector3& r)const;

		const float*		getHistogram() const {return histogram;};

		float				minDensity;
		float				maxDensity;
		float				avDensity;

		float				isoValue;
		float				threshold;
		int					isoBin;
		void 				updateBin();

		int					isOk() const {return ok;}
};
