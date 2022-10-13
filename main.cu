#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#include "blst_377_ops.h"
#include "types.h"
#include "asm_cuda.h"
#include "curand.h"

static const uint32_t WINDOW_SIZE = 128;
// static const uint32_t BLST_WIDTH = 253;

__global__ void msm6_pixel(blst_p1* bucket_lists, const blst_p1_affine* bases_in, const blst_scalar* scalars, const uint32_t* window_lengths, const uint32_t window_count) {
    limb_t index = threadIdx.x / 64;
    size_t shift = threadIdx.x - (index * 64);
    limb_t mask = (limb_t) 1 << (limb_t) shift;

    blst_p1 bucket;
    memcpy(&bucket, &BLS12_377_ZERO_PROJECTIVE, sizeof(blst_p1));

    uint32_t window_start = WINDOW_SIZE * blockIdx.x;
    uint32_t window_end = window_start + window_lengths[blockIdx.x];

    uint32_t activated_bases[WINDOW_SIZE];
    uint32_t activated_base_index = 0;

    // we delay the actual additions to a second loop because it reduces warp divergence (20% practical gain)
    for (uint32_t i = window_start; i < window_end; ++i) {
        limb_t bit = (scalars[i][index] & mask);
        if (bit == 0) {
            continue;
        }
        activated_bases[activated_base_index++] = i;
    }
    uint32_t i = 0;
    for (; i < (activated_base_index / 2 * 2); i += 2) {
        blst_p1 intermediate;
        blst_p1_add_affines_into_projective(&intermediate, &bases_in[activated_bases[i]], &bases_in[activated_bases[i + 1]]);
        blst_p1_add_projective_to_projective(&bucket, &bucket, &intermediate);
    }
    for (; i < activated_base_index; ++i) {
        blst_p1_add_affine_to_projective(&bucket, &bucket, &(bases_in[activated_bases[i]]));
    }

    memcpy(&bucket_lists[threadIdx.x * window_count + blockIdx.x], &bucket, sizeof(blst_p1));
}

__global__ void msm6_collapse_rows(blst_p1* target, const blst_p1* bucket_lists, const uint32_t window_count) {
    blst_p1 temp_target;
    uint32_t base = threadIdx.x * window_count;
    uint32_t term = base + window_count;
    memcpy(&temp_target, &bucket_lists[base], sizeof(blst_p1));

    for (uint32_t i = base + 1; i < term; ++i) {
        blst_p1_add_projective_to_projective(&temp_target, &temp_target, &bucket_lists[i]);
    }

    memcpy(&target[threadIdx.x], &temp_target, sizeof(blst_p1));
}


int main(void)
{



    int DATA_SIZE = 20000;

    thrust::device_vector<blst_p1_affine> bases(DATA_SIZE);
    blst_p1_affine b1 = {
            {
                    TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
                    TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
                    TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x1ae3a4617c510ea)
            },
            {
                    TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
                    TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
                    TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x1ae3a4617c510ea)
            }
    };
    for(int i=0; i< DATA_SIZE; i++){
        bases[i] = b1;
    }

    limb_t scalar_datas[DATA_SIZE][4] ;
    for (int i = 0; i<DATA_SIZE; i++) {
        for(int j=0; j<4; i++){
            scalar_datas[i][j] = TO_LIMB_T(0x1a22d9f300f5138f);
        }
    }
    blst_scalar *scalar_data;
    cudaMalloc(&scalar_data, sizeof(limb_t) * DATA_SIZE*4);
    cudaMemcpyAsync(scalar_data, scalar_datas, sizeof(limb_t) * 12, cudaMemcpyHostToDevice);


    uint32_t counts[]={1,2};
    int WINDOW_SIZE = 600;
    int window_count = (DATA_SIZE + WINDOW_SIZE -1)/WINDOW_SIZE;
    thrust::device_vector<blst_p1> bucket(253 * window_count);
    thrust::device_vector<blst_p1> result(253 );

    blst_p1 zero = {
            0,0
    };
    for(int i=0; i<253*window_count; i++) bucket[i] = zero;
    for(int i=0; i<253; i++) result[i] = zero;



    thrust::device_vector<uint32_t> windows(window_count);
    for (int i=0; i<window_count-1; i++) windows[i]=253;
    windows[window_count-1]=DATA_SIZE-DATA_SIZE/WINDOW_SIZE*253;


    msm6_pixel<<<window_count,253>>>(bucket.data().get(),bases.data().get(), scalar_data, windows.data().get(), window_count );
    msm6_collapse_rows<<<2,3>>>(result.data().get(), bucket.data().get(),  window_count);
    // H and D are automatically deleted when the function returns
    return 0;
}

/*
#include <stdio.h>
#include <cuda_runtime.h>

#include "curand.h"
#include <random>
#include "cxtimers.h"
#include "cx.h"
//#include <helper_functions.h>

/*long long int sum_part1(int seed, int points, int passes){
    int thread = omp_get_thread_num();
    std::default_random_engine gen(seed+113*thread);
    std::uniform_int_distribution<int> idist(0,2147483647);
    double idist_scale = 1.0/2147483647.0;

    long long int pisum = 0;
    for(int n=0; n< passes; n++){
        int subtot = 0;
        for(int k=0; k<points; k++){
            float x = idist_scale*idist(gen);
            float y = idist_scale*idist(gen);
            if(x*x+y*y<1.0f) subtot++;
        }
        pisum += subtot;
    }
    return pisum;
}*/
/*
void sum_part(float *rnum, int points, long long &pisum){
    unsigned int sum = 0;
    for (int i=0; i< points; i++){
        float x = rnum[i*2];
        float y = rnum[i*2 + 1];
        if(x*x+y*y<1.0f) sum++;
    }
    pisum+=sum;
}

int main(int argc, char *argv[]){
    std::random_device rd;
    int points = 1000000;
    int passes = (argc > 1) ? atoi (argv[1]) :1;

    int seed = (argc > 2)? atoi(argv[2]): rd();
    int bsize = points * 2 *sizeof(float);
    float *a; cudaMallocHost(&a, bsize);
    float *b; cudaMallocHost(&b, bsize);
    float *dev_rdm; cudaMalloc(&dev_rdm, bsize);

    cudaEvent_t copydone; cudaEventCreate(&copydone);

    long long int pisum = 0;
    cx::timer tim;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateUniform(gen, dev_rdm, points*2);
    cudaMemcpy(a, dev_rdm, bsize, cudaMemcpyDeviceToHost);
    for(int k=0; k<passes; k++){
        curandGenerateUniform(gen, dev_rdm, points*2);
        cudaMemcpyAsync(b, dev_rdm, bsize, cudaMemcpyDeviceToHost);
        cudaEventRecord(copydone, 0);
        cudaEventQuery(copydone);
        sum_part(a, points, pisum);
        std::swap(a, b);
        cudaStreamWaitEvent(0, copydone, 0);
    }

    double t1 = tim.lap_ms();
    double pi = 4.0*(double)pisum/((double)points*(double)passes);
    long long ntot = passes*points;
    double frac_error = 1000000.0*(pi - cx::pi<double>)/cx::pi<double>;
    printf("pi = %10.8f err %.1f,ntot %lld, time %.3fms\n", pi, frac_error, ntot, t1);

    cudaFreeHost(a); cudaFreeHost(b);cudaFree(dev_rdm);
    curandDestroyGenerator(gen);
    return 0;
}
*/



