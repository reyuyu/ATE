#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <new>
struct Plan {
 cublasLtHandle_t h=nullptr;
 cublasLtMatmulDesc_t op=nullptr;
 cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
 cublasLtMatmulPreference_t pref=nullptr;
 cublasLtMatmulHeuristicResult_t choices[2]{};
 cublasLtMatmulAlgo_t algo{};
 int count=0,chosen=-1;
 ~Plan(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(op)cublasLtMatmulDescDestroy(op);if(h)cublasLtDestroy(h);}
};
extern "C" {
uint64_t lt_version(){return cublasLtGetVersion();}
void* lt_plan(int m,int n,int k,size_t workspace,int* status){
 Plan* p=new(std::nothrow) Plan();if(!p){*status=-99;return nullptr;}
 #define TRY(expr) do{auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS){*status=int(s);delete p;return nullptr;}}while(0)
 TRY(cublasLtCreate(&p->h));TRY(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
 cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
 TRY(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));TRY(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
 // W[N,K] and X[M,K] row-major correspond to column-major A[K,N], B[K,M].
 TRY(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16BF,k,n,k));TRY(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16BF,k,m,k));TRY(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n));
 TRY(cublasLtMatmulPreferenceCreate(&p->pref));TRY(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
 TRY(cublasLtMatmulAlgoGetHeuristic(p->h,p->op,p->a,p->b,p->c,p->c,p->pref,2,p->choices,&p->count));
 if(p->count>0){p->algo=p->choices[0].algo;p->chosen=0;}*status=0;return p;
 #undef TRY
}
int lt_count(void* x){return static_cast<Plan*>(x)->count;}
int lt_select(void* dst,void* src,int index,size_t workspace){
 Plan* d=static_cast<Plan*>(dst);Plan* s=static_cast<Plan*>(src);if(index<0||index>=s->count)return -98;
 cublasLtMatmulHeuristicResult_t result{};auto r=cublasLtMatmulAlgoCheck(d->h,d->op,d->a,d->b,d->c,d->c,&s->choices[index].algo,&result);
 if(r!=CUBLAS_STATUS_SUCCESS)return int(r);if(result.state!=CUBLAS_STATUS_SUCCESS)return int(result.state);if(result.workspaceSize>workspace)return -97;
 d->algo=s->choices[index].algo;d->chosen=index;return 0;
}
int lt_attr(void* x,int attribute,int* value){
 size_t written=0;auto* p=static_cast<Plan*>(x);return int(cublasLtMatmulAlgoConfigGetAttribute(&p->algo,static_cast<cublasLtMatmulAlgoConfigAttributes_t>(attribute),value,sizeof(*value),&written));
}
int lt_run(void* x,void* w,void* input,void* output,void* workspace,size_t size,void* stream){
 auto* p=static_cast<Plan*>(x);if(p->chosen<0)return -96;float alpha=1.0f,beta=0.0f;
 return int(cublasLtMatmul(p->h,p->op,&alpha,w,p->a,input,p->b,&beta,output,p->c,output,p->c,&p->algo,workspace,size,reinterpret_cast<cudaStream_t>(stream)));
}
void lt_destroy(void* x){delete static_cast<Plan*>(x);}
}
