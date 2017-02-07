#include <cMatrix44.h>
cMatrix44::cMatrix44(double mat[4][4]){
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)(*this)(i,j)=mat[i][j];
}
cMatrix44::cMatrix44(const cMatrix33 &rot, const cVector3 &shift){
    for(int i=0;i<3;i++){
        (*this)(3,i)=shift[i];
        for(int j=0;j<3;j++)
            (*this)(i,j)=rot(i,j);
    }
    (*this)(4,4)=1;
}

cMatrix44 cMatrix44::operator*(const cMatrix44 &mat) const{
    double res[4][4];
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++){
            res[i][j]=0.0;
            for(int k=0;k<4;k++)
                res[i][j]+=(*this)(i,k)*mat(k,j);
        }
    return cMatrix44(res);
}

cVector3 cMatrix44::operator*(const cVector3 &vec) const{
    double vec4[4];
    vec4[0]=vec[0];vec4[1]=vec[1];vec4[2]=vec[2];vec4[3]=1.0;

    for(int i=0;i<3;i++){
        vec4[i]=0;
        for(int j=0;j<4;j++)
            vec4[i]+=(*this)(i,j)*vec4[j];
    }
    return cVector3(vec4[0], vec4[1], vec4[2]);
}

void cMatrix44::print()	{

	double SMALL_VALUE=10e-200;
	for (int i=0;i<4;i++) {

		for (int j=0;j<4;j++) {
			if (fabs(m[i][j]) < SMALL_VALUE && fabs(m[i][j]) > 0.0)
				std::cout << "\t[" << "O" << "]";
			else {
				std::cout << "\t[" << m[i][j] << "]";
			}
		}
		std::cout << std::endl;

	}

	std::cout << std::endl;
	std::cout << std::endl;

}