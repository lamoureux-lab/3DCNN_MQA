local ffi = require 'ffi'

ffi.cdef[[
typedef struct{ char **strings; size_t len; } batchInfo;
void deleteBatchInfo(batchInfo* binfo);
void printBatchInfo(batchInfo* binfo);
void pushStringBatchInfo(const char* string, batchInfo* binfo);

]]
local C = ffi.load'../build/libtest_lua_bindings.so'




for i=1,10 do 
	ffi.C.pushStringBatchInfo("aaaaa"..tostring(i), batchInfo)
end
ffi.C.printBatchInfo(batchInfo)

