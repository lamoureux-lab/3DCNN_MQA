#include <iostream>
#include <string>

extern "C" {
	typedef struct{
		char **strings;
		size_t len;
	} batchInfo;

	batchInfo* createBatchInfo(int batch_size){
		batchInfo *binfo;
		binfo = new batchInfo;
		binfo->strings = new char*[batch_size];
		binfo->len = batch_size;
		return binfo;
	}

	void deleteBatchInfo(batchInfo* binfo){
		for(int i=0;i<binfo->len;i++){
			delete [] binfo->strings[i];
		}
		binfo->len=0;
		delete binfo;
	}

	void pushStringBatchInfo(const char* string, batchInfo* binfo){
		std::string str(string);

		binfo->strings[binfo->len] = new char[str.length()];
		for(int i=0; i<str.length(); i++){
			binfo->strings[binfo->len][i] = str[i];
		}

	}

	void printBatchInfo(batchInfo *binfo){
		for(int i=0;i<binfo->len;i++){
			std::cout<<binfo->strings[i]<<"\n";
		}
	}
}