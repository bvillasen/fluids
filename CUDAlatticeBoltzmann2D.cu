// #include <pycuda-complex.hpp>
// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>

#define I2D(ni,i,j) (((ni)*(j)) + i)
#define T0_f(t_j,nj) (tempMax + t_j*(tempMin-tempMax)/(nj-1))
// #define pi 3.14159265f

// textures on device //
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_f1, tex_f2, tex_f3, tex_f4, tex_f5, tex_f6, tex_f7, tex_f8;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_g1, tex_g2, tex_g3, tex_g4, tex_g5, tex_g6, tex_g7, tex_g8;


////////////////////////////////////////////////////////////////////////////////
//////////////////////          STREAM                //////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void stream_kernel ( float *f1_in, float *f2_in, float *f3_in, float *f4_in, float *f5_in, float *f6_in, float *f7_in, float *f8_in,
                                float *g1_in, float *g2_in, float *g3_in, float *g4_in, float *g5_in, float *g6_in, float *g7_in, float *g8_in){
  
  int t_i = blockIdx.x*blockDim.x + threadIdx.x;
  int t_j = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_i + t_j*blockDim.x*gridDim.x;
  
  // look up the adjacent f's needed for streaming using textures
  // i.e. gather from textures, write to device memory: f1_in, etc
  f1_in[tid] = tex2D(tex_f1, (float) (t_i-1)  , (float) t_j);
  f2_in[tid] = tex2D(tex_f2, (float) t_i      , (float) (t_j-1));
  f3_in[tid] = tex2D(tex_f3, (float) (t_i+1)  , (float) t_j);
  f4_in[tid] = tex2D(tex_f4, (float) t_i      , (float) (t_j+1));
  f5_in[tid] = tex2D(tex_f5, (float) (t_i-1)  , (float) (t_j-1));
  f6_in[tid] = tex2D(tex_f6, (float) (t_i+1)  , (float) (t_j-1));
  f7_in[tid] = tex2D(tex_f7, (float) (t_i+1)  , (float) (t_j+1));
  f8_in[tid] = tex2D(tex_f8, (float) (t_i-1)  , (float) (t_j+1));
  
  g1_in[tid] = tex2D(tex_g1, (float) (t_i-1)  , (float) t_j);
  g2_in[tid] = tex2D(tex_g2, (float) t_i      , (float) (t_j-1));
  g3_in[tid] = tex2D(tex_g3, (float) (t_i+1)  , (float) t_j);
  g4_in[tid] = tex2D(tex_g4, (float) t_i      , (float) (t_j+1));
  g5_in[tid] = tex2D(tex_g5, (float) (t_i-1)  , (float) (t_j-1));
  g6_in[tid] = tex2D(tex_g6, (float) (t_i+1)  , (float) (t_j-1));
  g7_in[tid] = tex2D(tex_g7, (float) (t_i+1)  , (float) (t_j+1));
  g8_in[tid] = tex2D(tex_g8, (float) (t_i-1)  , (float) (t_j+1));
}


////////////////////////////////////////////////////////////////////////////////
//////////////////////          COLLIDE               //////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void collide_kernel (int ni, int nj, float beta, float tau, float tauT, float faceq1, float faceq2, float faceq3, 
				float *f0_in, float *f1_in, float *f2_in, float *f3_in, float *f4_in, float *f5_in, float *f6_in, float *f7_in, float *f8_in,
				float *g0_in, float *g1_in, float *g2_in, float *g3_in, float *g4_in, float *g5_in, float *g6_in, float *g7_in, float *g8_in,
				float g, float tempMin, float tempMax, int plotTemp,  float *plot_data){
  int t_i = blockIdx.x*blockDim.x + threadIdx.x;
  int t_j = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_i + t_j*blockDim.x*gridDim.x;

  float rtau = 1.f/tau;
  float rtau1 = 1.f - rtau;   
  
  float rtauT = 1.f/tauT;
  float rtau1T = 1.f - rtauT; 

  // Read all f's and g's and store in registers
  float f0now = f0_in[tid];
  float f1now = f1_in[tid];
  float f2now = f2_in[tid];
  float f3now = f3_in[tid];
  float f4now = f4_in[tid];
  float f5now = f5_in[tid];
  float f6now = f6_in[tid];
  float f7now = f7_in[tid];
  float f8now = f8_in[tid];
  
  float g0now = g0_in[tid];
  float g1now = g1_in[tid];
  float g2now = g2_in[tid];
  float g3now = g3_in[tid];
  float g4now = g4_in[tid];
  float g5now = g5_in[tid];
  float g6now = g6_in[tid];
  float g7now = g7_in[tid];
  float g8now = g8_in[tid];

  // Macroscopic flow props:
  float ro =  f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
  float T  =  g0now + g1now + g2now + g3now + g4now + g5now + g6now + g7now + g8now;
  float vx = (f1now - f3now + f5now - f6now - f7now + f8now)/ro;
  float vy = (f2now - f4now + f5now + f6now - f7now - f8now)/ro;

  // Set plotting variable to velocity magnitude
  if (plotTemp == 1) plot_data[tid] = T;
/*  else plot_data[tid] = 0.3f*sqrtf(vx*vx + vy*vy);*/
  else plot_data[tid] = 0.2f*vy;
//   plot_data[tid] = T;
  // Calculate equilibrium f's
  float v_sq_term = 1.5f*(vx*vx + vy*vy);
  float f0eq = ro * faceq1 * (1.f - v_sq_term);
  float f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
  float f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
  float f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
  float f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
  float f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
  float f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
  float f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
  float f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);
  
    // Calculate equilibrium g's
//   float g0eq = T * faceq1 * (1.f );
//   float g1eq = T * faceq2 * (1.f + 3.f*vx );
//   float g2eq = T * faceq2 * (1.f + 3.f*vy );
//   float g3eq = T * faceq2 * (1.f - 3.f*vx );
//   float g4eq = T * faceq2 * (1.f - 3.f*vy );
//   float g5eq = T * faceq3 * (1.f + 3.f*(vx + vy) );
//   float g6eq = T * faceq3 * (1.f + 3.f*(-vx + vy) );
//   float g7eq = T * faceq3 * (1.f + 3.f*(-vx - vy) );
//   float g8eq = T * faceq3 * (1.f + 3.f*(vx - vy) );
  
  float g0eq = T * faceq1 * (1.f - v_sq_term);
  float g1eq = T * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
  float g2eq = T * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
  float g3eq = T * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
  float g4eq = T * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
  float g5eq = T * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
  float g6eq = T * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
  float g7eq = T * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
  float g8eq = T * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

  // Do collisions (no temperature effect)
//   f0_in[tid] = rtau1 * f0now + rtau * f0eq;
//   f1_in[tid] = rtau1 * f1now + rtau * f1eq;
//   f2_in[tid] = rtau1 * f2now + rtau * f2eq;
//   f3_in[tid] = rtau1 * f3now + rtau * f3eq;
//   f4_in[tid] = rtau1 * f4now + rtau * f4eq;
//   f5_in[tid] = rtau1 * f5now + rtau * f5eq;
//   f6_in[tid] = rtau1 * f6now + rtau * f6eq;
//   f7_in[tid] = rtau1 * f7now + rtau * f7eq;
//   f8_in[tid] = rtau1 * f8now + rtau * f8eq;
  
  float T0 = T0_f(t_j,nj);
  //Apply temperature gradient effect
  f0_in[tid] = rtau1 * f0now + rtau * f0eq;
  f1_in[tid] = rtau1 * f1now + rtau * f1eq;
  f2_in[tid] = rtau1 * f2now + rtau * f2eq - 3*beta*faceq2*(T-T0)*g;
  f3_in[tid] = rtau1 * f3now + rtau * f3eq;
  f4_in[tid] = rtau1 * f4now + rtau * f4eq + 3*beta*faceq2*(T-T0)*g;
  f5_in[tid] = rtau1 * f5now + rtau * f5eq - 3*beta*faceq3*(T-T0)*g;
  f6_in[tid] = rtau1 * f6now + rtau * f6eq - 3*beta*faceq3*(T-T0)*g;
  f7_in[tid] = rtau1 * f7now + rtau * f7eq + 3*beta*faceq3*(T-T0)*g;
  f8_in[tid] = rtau1 * f8now + rtau * f8eq + 3*beta*faceq3*(T-T0)*g;
  
  g0_in[tid] = rtau1T * g0now + rtauT * g0eq;
  g1_in[tid] = rtau1T * g1now + rtauT * g1eq;
  g2_in[tid] = rtau1T * g2now + rtauT * g2eq;
  g3_in[tid] = rtau1T * g3now + rtauT * g3eq;
  g4_in[tid] = rtau1T * g4now + rtauT * g4eq;
  g5_in[tid] = rtau1T * g5now + rtauT * g5eq;
  g6_in[tid] = rtau1T * g6now + rtauT * g6eq;
  g7_in[tid] = rtau1T * g7now + rtauT * g7eq;
  g8_in[tid] = rtau1T * g8now + rtauT * g8eq;
  

}
////////////////////////////////////////////////////////////////////////////////
//////////////////////          APPLY BC              //////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void apply_BCs_kernel (int ni, int nj, float vxin, float roout, float faceq1, float faceq2, float faceq3, 
				  float *f0_in, float *f1_in, float *f2_in, float *f3_in, float *f4_in, float *f5_in, float *f6_in, float *f7_in, float *f8_in,
				  float *g0_in, float *g1_in, float *g2_in, float *g3_in, float *g4_in, float *g5_in, float *g6_in, float *g7_in, float *g8_in, 
				  float tempMin, float tempMax, int* solid_data){
  // Apply all BC's apart from periodic boundaries:
  int t_i = blockIdx.x*blockDim.x + threadIdx.x;
  int t_j = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_i + t_j*blockDim.x*gridDim.x;

  // Solid BC: "bounce-back"
  if (solid_data[tid] == 0) {
    float f1old = f1_in[tid];
    float f2old = f2_in[tid];
    float f3old = f3_in[tid];
    float f4old = f4_in[tid];
    float f5old = f5_in[tid];
    float f6old = f6_in[tid];
    float f7old = f7_in[tid];
    float f8old = f8_in[tid];
    
    f1_in[tid] = f3old;
    f2_in[tid] = f4old;
    f3_in[tid] = f1old;
    f4_in[tid] = f2old;
    f5_in[tid] = f7old;
    f6_in[tid] = f8old;
    f7_in[tid] = f5old;
    f8_in[tid] = f6old;
  }
  
  if (t_j== 0) {
    float f1old = f1_in[tid];
    float f2old = f2_in[tid];
    float f3old = f3_in[tid];
    float f4old = f4_in[tid];
    float f5old = f5_in[tid];
    float f6old = f6_in[tid];
    float f7old = f7_in[tid];
    float f8old = f8_in[tid];
    
    f1_in[tid] = f3old;
    f2_in[tid] = f4old;
    f3_in[tid] = f1old;
    f4_in[tid] = f2old;
    f5_in[tid] = f7old;
    f6_in[tid] = f8old;
    f7_in[tid] = f5old;
    f8_in[tid] = f6old;
    
    g0_in[tid] = faceq1 * tempMax;
    g1_in[tid] = faceq2 * tempMax;
    g2_in[tid] = faceq2 * tempMax;
    g3_in[tid] = faceq2 * tempMax;
    g4_in[tid] = faceq2 * tempMax;
    g5_in[tid] = faceq3 * tempMax;
    g6_in[tid] = faceq3 * tempMax;
    g7_in[tid] = faceq3 * tempMax;
    g8_in[tid] = faceq3 * tempMax;
}

  if (t_j== nj-1 ) {
    float f1old = f1_in[tid];
    float f2old = f2_in[tid];
    float f3old = f3_in[tid];
    float f4old = f4_in[tid];
    float f5old = f5_in[tid];
    float f6old = f6_in[tid];
    float f7old = f7_in[tid];
    float f8old = f8_in[tid];
    
    f1_in[tid] = f3old;
    f2_in[tid] = f4old;
    f3_in[tid] = f1old;
    f4_in[tid] = f2old;
    f5_in[tid] = f7old;
    f6_in[tid] = f8old;
    f7_in[tid] = f5old;
    f8_in[tid] = f6old;
    
    g0_in[tid] = faceq1 * tempMin;
    g1_in[tid] = faceq2 * tempMin;
    g2_in[tid] = faceq2 * tempMin;
    g3_in[tid] = faceq2 * tempMin;
    g4_in[tid] = faceq2 * tempMin;
    g5_in[tid] = faceq3 * tempMin;
    g6_in[tid] = faceq3 * tempMin;
    g7_in[tid] = faceq3 * tempMin;
    g8_in[tid] = faceq3 * tempMin;
  }
  
    if (t_j== 1) {
    g0_in[tid] = faceq1 * tempMax;
    g1_in[tid] = faceq2 * tempMax;
    g2_in[tid] = faceq2 * tempMax;
    g3_in[tid] = faceq2 * tempMax;
    g4_in[tid] = faceq2 * tempMax;
    g5_in[tid] = faceq3 * tempMax;
    g6_in[tid] = faceq3 * tempMax;
    g7_in[tid] = faceq3 * tempMax;
    g8_in[tid] = faceq3 * tempMax;
}
//   if (t_j== nj-2 ) {
//     g0_in[tid] = faceq1 * tempMin;
//     g1_in[tid] = faceq2 * tempMin;
//     g2_in[tid] = faceq2 * tempMin;
//     g3_in[tid] = faceq2 * tempMin;
//     g4_in[tid] = faceq2 * tempMin;
//     g5_in[tid] = faceq3 * tempMin;
//     g6_in[tid] = faceq3 * tempMin;
//     g7_in[tid] = faceq3 * tempMin;
//     g8_in[tid] = faceq3 * tempMin;
//   }


//   // Inlet BC - very crude
//   if (t_i == 0) {
//     float v_sq_term = 1.5f*(vxin * vxin);
//     
//     f1_in[tid] = roout * faceq2 * (1.f + 3.f*vxin + 3.f*v_sq_term);
//     f5_in[tid] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
//     f8_in[tid] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
// 
//   }
//       
//   // Exit BC - very crude
//   if (t_i == (ni-1)) {
//     int tid_2 = tid - 1;
//     f3_in[tid] = f3_in[tid_2];
//     f6_in[tid] = f6_in[tid_2];
//     f7_in[tid] = f7_in[tid_2];
// 
//   }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void apply_Periodic_BC_kernel (int ni, int nj, 
					  float *f1_in, float *f2_in, float *f3_in, float *f4_in, float *f5_in, float *f6_in, float *f7_in, float *f8_in,
					  float *g1_in, float *g2_in, float *g3_in, float *g4_in, float *g5_in, float *g6_in, float *g7_in, float *g8_in){
  int t_i = blockIdx.x*blockDim.x + threadIdx.x;
  int t_j = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_i + t_j*blockDim.x*gridDim.x;

  int tid_2;
  
  //periodicas en tapas de los lados
 if (t_i == 0 ) {
      tid_2 = tid + ni -1;
      f1_in[tid] = f1_in[tid_2];
      f5_in[tid] = f5_in[tid_2];
      f8_in[tid] = f8_in[tid_2];
      
      g1_in[tid] = g1_in[tid_2];
      g5_in[tid] = g5_in[tid_2];
      g8_in[tid] = g8_in[tid_2];
  }
  if (t_i == (ni-1)) {
      tid_2 = tid - ni +1;
      f3_in[tid] = f3_in[tid_2];
      f6_in[tid] = f6_in[tid_2];
      f7_in[tid] = f7_in[tid_2];
      
      g3_in[tid] = g3_in[tid_2];
      g6_in[tid] = g6_in[tid_2];
      g7_in[tid] = g7_in[tid_2];
  }
  
  
  
  //Periodicas en tapas de Arriba
//   if (t_j == 0 ) {
//       tid_2 = t_i + (nj-1)*pitch/sizeof(float);
//       f2_in[tid] = f2_in[tid_2];
//       f5_in[tid] = f5_in[tid_2];
//       f6_in[tid] = f6_in[tid_2];
//   }
//   if (t_j == (nj-1)) {
//       tid_2 = t_i;
//       f4_in[tid] = f4_in[tid_2];
//       f7_in[tid] = f7_in[tid_2];
//       f8_in[tid] = f8_in[tid_2];
//   }
}
