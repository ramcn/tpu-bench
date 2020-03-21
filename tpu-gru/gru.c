#include <math.h>
#include "layers.h"
#include "flappie_stdlib.h"
#include <sys/time.h>

#    define LOGISTICF gpu_logisticf 
#    define TANHF gpu_logisticf

#    define _A 12102203.161561485f
#    define _B 1065353216.0f
#    define _BOUND 88.02969193111305

static inline float max(float a, float b){
	if (a > b)
		return a;
	else
		return b;
}

static inline float min(float a, float b){
	if (a < b)
		return a;
	else
		return b;
}

static inline float gpu_expf(float x) {
    x = max(-_BOUND, min(_BOUND, x));
    union {
        uint32_t i;
        float f;
    } value = {
    .i = (uint32_t) (_A * x + _B)};
    return value.f;
}

static inline float gpu_logisticf(float x) {
    return 1.0 / (1.0 + gpu_expf(-x));
}

static inline float gpu_tanhf(float x) {
    const float y = gpu_logisticf(x + x);
    return y + y - 1.0;
}


struct timeval start, end_time;
long useconds, seconds, mseconds;

void mat_mul_c(float *a_rm, float *b_cm, float *c_in, float *c_out,
                  uint32_t M, uint32_t N, uint32_t P,
                  bool a_trans, bool b_trans, float alpha, float beta,
                  uint32_t a_stride, uint32_t b_stride, uint32_t c_stride)
{
    uint32_t a_inner_stride, a_outer_stride, b_inner_stride, b_outer_stride;
    if(a_trans) {
        a_inner_stride = a_stride; a_outer_stride = 1;
    } else {
        a_inner_stride = 1;        a_outer_stride = a_stride;
    }
    if(b_trans) {
        b_inner_stride = b_stride; b_outer_stride = 1;
    } else {
        b_inner_stride = 1;        b_outer_stride = b_stride;
    }

    for(uint32_t i = 0; i < M; i++) {
        for(uint32_t j = 0; j < P; j++) {
            float acc = 0;
            const float *a  = a_rm + (a_outer_stride * i); // row selection
            const float *b  = b_cm + (b_inner_stride * j); // col selection
            for(uint32_t k = 0; k < N; k++) {
                acc += a[k * a_inner_stride] * b[k * b_outer_stride];
            }
            uint32_t idx = (c_stride * i) + j;
            c_out[idx] = (beta * c_in[idx]) + (alpha * acc);
        }
    }
}

void grumod_step(const_flappie_matrix x, const_flappie_matrix istate,
                 const_flappie_matrix sW, flappie_matrix xF,
                 flappie_matrix ostate) {
    /* Perform a single modified GRU step
     * x      is [isize]
     * istate is [size]
     * xW     is [isize, 3 * size]
     * sW     is [size, 2 * size]
     * sW2    is [size, size]
     * bias   is [3 * size]
     * xF     is [3 * size]
     * ostate is [size]
     */
    assert(NULL != x);
    assert(NULL != sW);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(3 * size == sW->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);

    // Copy input vector = iW x + b to temporary vector and zero last chunk
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    memset(xF->data.v + sizeq + sizeq, 0, sizeq *sizeof(__m128));
    /*  Add sW * istate to first 3 * size elts of xF
     *  then apply gate function to get r and z
     */
    //cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
               // sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);
    mat_mul_c(sW->data.f, istate->data.f,xF->data.f,xF->data.f, 768, 256, 1, 0,0, 1.0, 0.0, 256, 1, 1);

    for (size_t i = 0; i < (size + size); i++) {
        xF->data.f[i] = LOGISTICF(xF->data.f[i]);
    }

    const float *z = xF->data.f;
    const float *a = xF->data.f + size;
    float *b = xF->data.f + size + size;
    float *c =  x->data.f + size + size;

    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] * b[i] + c[i]; // cin and cout are same and below tanh will be done in place
    }


    for (size_t i = 0; i < size; i++) {
        c[i] = TANHF(c[i]);
    }

    const float ones = 1.0f;
    float *c1 = ostate->data.f;

    for (size_t i = 0; i < size ; i++) {
        c1[i] = (-1) * z[i] * c[i] + c[i]; // cin and cout are different. alpha is -1.
        c1[i] = z[i] * istate->data.f[i] + c1[i]; // cin and cout are same.
    }

    /*for (size_t i = 0; i < (sizeq + sizeq); i++) {
        xF->data.v[i] = LOGISTICFV(xF->data.v[i]);
    }

    const __m128 *z = xF->data.v;
    const __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = r[i] * hbar[i] + x->data.v[sizeq + sizeq + i];
    }
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = TANHFV(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    } */
}


int tpu_mat_mul(int iteration);

flappie_matrix aes_grumod_linear_cpu_gpu( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b, int layer, flappie_matrix Xnext, flappie_matrix xColTmp, flappie_matrix xCol) {  
    assert(NULL != sW);

    const size_t size = sW->nr;
    const size_t N = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    _Mat XnextBuf; 
    float Cin[768], Cout[768];
    float *ostate_ptr;
    float *istate_ptr;

    float al =1.0f;
    float bet =1.0f;
    for (size_t c = 0; c < Xnext->nc; c++) {
        memcpy(Xnext->data.v + c * Xnext->nrq, b->data.v, Xnext->nrq * sizeof(__m128));
    }

    for (int i = 1; i < N; i++) {
        size_t index, index2;
        // LOAD
        {
                if(backward) {
                        index = N - i - 1;
                        xCol->data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr + 256;
                        XnextBuf.data.f = Xnext->data.f + (index+1) * Xnext->nr;
			index2 = index + 1;
                }

                else {
                        index = i;
                        xCol->data.f = X->data.f + index * X->nr;
                        ostate_ptr = ostate->data.f + index * ostate->nr;
                        istate_ptr = ostate_ptr - 256;
                        XnextBuf.data.f = Xnext->data.f + (index-1) * Xnext->nr;
			index2 = index - 1; 
                }
        }

        // COMPUTE
        {
                const size_t size = 256;
    		    int M=768, N=256;

                memcpy(Cin, xCol->data.f, 768*sizeof(float));
                memcpy(Cout, xColTmp->data.f, 768*sizeof(float));
                memcpy(Cout, Cin, 768 * sizeof(float) );
                memset(Cout + size + size, 0, size *sizeof(float));

                //gettimeofday(&start, NULL);
                //cblas_sgemv(CblasRowMajor, CblasNoTrans, 768, 256, 1.0, sW->data.f, 256, istate_ptr, 1, 1.0, Cout, 1);
                //mat_mul_c(sW->data.f, istate_ptr,Cin,Cout, 768, 256, 1, 0,0, 1.0, 0.0, 256, 1, 1);
                tpu_mat_mul(i);
                tpu_mat_mul(i);
                tpu_mat_mul(i);

                if(i%100 == 0)
                	printf("Successfully completed %d iteration \n", i);

                for (size_t i = 0; i < size; i++) {
                        Cout[i] = LOGISTICF(Cout[i]);
                        Cout[size+i] = LOGISTICF(Cout[size+i]);
                        Cout[i+size+size] = TANHF(Cout[i+size] * Cout[i+size+size] + Cin[i+size+size]);
                        ostate_ptr[i] = (-1) * Cout[i] * Cout[i+size+size] + Cout[i+size+size];
                        ostate_ptr[i] = Cout[i] * istate_ptr[i] + ostate_ptr[i];
                }
                //cblas_sgemv(CblasRowMajor, CblasNoTrans, W->nc, W->nr, 1.0, W->data.f, W->stride, ostate_ptr, 1, 1.0, XnextBuf.data.f, 1);
                //mat_mul_c(W->data.f, ostate_ptr,XnextBuf.data.f,XnextBuf.data.f, 768, 256, 1, 0,0, 1.0, 0.0, 256, 1, 1);
                tpu_mat_mul(i);
                tpu_mat_mul(i);
                tpu_mat_mul(i);
    		//gettimeofday(&end_time, NULL);
    		useconds = end_time.tv_usec - start.tv_usec;
    		seconds = end_time.tv_sec - start.tv_sec;
   	 	mseconds += ((seconds) * 1000 + useconds/1000.0) + 0.5;
        }
    } // end of N iterations

    return Xnext;
}


flappie_matrix aes_grumod_linear_gpu_wrapper( const_flappie_matrix X, const_flappie_matrix sW, flappie_matrix ostate, int backward, const_flappie_matrix W, const_flappie_matrix b, int layer) {

    const size_t size = sW->nr;
    const size_t N = X->nc;
    flappie_matrix xColTmp;
    flappie_matrix Xnext;
    ostate = remake_flappie_matrix(ostate, size, N);
    xColTmp = make_flappie_matrix(3 * size, 1);
    Xnext = remake_flappie_matrix(NULL, W->nc, ostate->nc);


    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));

    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;

    if(backward) {
      xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
      sCol1.data.v = ostate->data.v;
      sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
      grumod_step(&xCol, &sCol1, sW, xColTmp, &sCol2);
    }
    else {
      sCol1.data.v = ostate->data.v + ostate->nrq;
      sCol2.data.v = ostate->data.v;
      grumod_step(&xCol, &sCol1, sW, xColTmp, &sCol2);
    }

    Xnext = aes_grumod_linear_cpu_gpu(X,sW,ostate,backward,W,b,layer,Xnext, xColTmp, &xCol);

    xColTmp = free_flappie_matrix(xColTmp);

    return Xnext;
}
