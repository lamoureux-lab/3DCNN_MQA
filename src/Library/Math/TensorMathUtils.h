#include <TH.h>
extern "C" void interpolateTensor(THFloatTensor *src, THFloatTensor *dst);
extern "C" void extrapolateTensor(THFloatTensor *src, THFloatTensor *dst);
extern "C" void forwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);
extern "C" void backwardGrad2Sum(THFloatTensor *src, THFloatTensor *dst);