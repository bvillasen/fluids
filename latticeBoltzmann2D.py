# 2D fluid simulation using Lattice-Boltzmann method 
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAlaticceBoltzmann2D.cu, animation2D.py, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/animation2D
#                               https://github.com/bvillasen/tools
import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
animation2DDirectory = parentDirectory + "/animation2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )

import animation2D
from cudaTools import setCudaDevice, getFreeMemory, gpuArray2DtocudaArray

#nPoints = 512*2
useDevice = None
for option in sys.argv:
  #if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
#set simulation volume dimentions 
nWidth = 128
nHeight = 64
nData = nWidth*nHeight

tempMin = 0.0001
tempMax = 0.05
beta = 0.001 
g = -40
vxIn =0.04
tau = 0.55
tauT = 0.55
rhoOut = 1.0

faceq1 = 4./9
faceq2 = 1./9
faceq3 = 1./36

plotVar = 1  #1->Temperature

#Convert parameters to float32
beta = np.float32( beta )
g = np.float32( g )
vxIn = np.float32( vxIn )
tau = np.float32( tau )
tauT = np.float32( tauT )
rhoOut = np.float32( rhoOut )
faceq1 = np.float32( faceq1 )
faceq2 = np.float32( faceq2 )
faceq3 = np.float32( faceq3 )
tempMax = np.float32(tempMax)
tempMin = np.float32(tempMin)
plotVar = np.int32( plotVar )

#Initialize openGL
animation2D.nWidth = nWidth
animation2D.nHeight = nHeight
animation2D.initGL()

#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
grid2D = (gridx, gridy, 1)
block2D = (block_size_x, block_size_y, 1)

#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "Compiling CUDA code"
cudaCodeString_raw = open("CUDAlatticeBoltzmann2D.cu", "r").read()
cudaCodeString = cudaCodeString_raw # % { "BLOCK_WIDTH":block2D[0], "BLOCK_HEIGHT":block2D[1], "BLOCK_DEPTH":block2D[2], }
cudaCode = SourceModule(cudaCodeString)
tex_f1 = cudaCode.get_texref('tex_f1')
tex_f2 = cudaCode.get_texref('tex_f2')
tex_f3 = cudaCode.get_texref('tex_f3')
tex_f4 = cudaCode.get_texref('tex_f4')
tex_f5 = cudaCode.get_texref('tex_f5')
tex_f6 = cudaCode.get_texref('tex_f6')
tex_f7 = cudaCode.get_texref('tex_f7')
tex_f8 = cudaCode.get_texref('tex_f8')
tex_g1 = cudaCode.get_texref('tex_g1')
tex_g2 = cudaCode.get_texref('tex_g2')
tex_g3 = cudaCode.get_texref('tex_g3')
tex_g4 = cudaCode.get_texref('tex_g4')
tex_g5 = cudaCode.get_texref('tex_g5')
tex_g6 = cudaCode.get_texref('tex_g6')
tex_g7 = cudaCode.get_texref('tex_g7')
tex_g8 = cudaCode.get_texref('tex_g8')
streamKernel = cudaCode.get_function('stream_kernel')
collideKernel = cudaCode.get_function('collide_kernel')
applyBCsKernel = cudaCode.get_function('apply_BCs_kernel')
applyPeriodicBCsKernel = cudaCode.get_function('apply_Periodic_BC_kernel')
#########################################################################
def initialGradient():
  a =[ [float(i)*(tempMin-tempMax)/(nHeight-1) + tempMax]*nWidth for i in range(nHeight) ]
  return np.array(a).astype(np.float32)
#########################################################################
def stream():
  copyf1_dTod( aligned=True )
  copyf2_dTod( aligned=True )
  copyf3_dTod( aligned=True )
  copyf4_dTod( aligned=True )
  copyf5_dTod( aligned=True )
  copyf6_dTod( aligned=True )
  copyf7_dTod( aligned=True )
  copyf8_dTod( aligned=True )
  
  copyg1_dTod( aligned=True )
  copyg2_dTod( aligned=True )
  copyg3_dTod( aligned=True )
  copyg4_dTod( aligned=True )
  copyg5_dTod( aligned=True )
  copyg6_dTod( aligned=True )
  copyg7_dTod( aligned=True )
  copyg8_dTod( aligned=True )
  streamKernel(f1_d, f2_d, f3_d, f4_d, f5_d, f6_d, f7_d, f8_d, 
	       g1_d, g2_d, g3_d, g4_d, g5_d, g6_d, g7_d, g8_d, grid=grid2D, block=block2D )
#########################################################################

  

#########################################################################
#def sendToScreen( plotData ):
  ##maxVal = gpuarray.max(plotData).get() + 0.00005
  ##multiplyByFloat( 1./maxVal, plotData )
  #floatToUchar( plotData, plotData_d )
  #copyToScreenArray()
#########################################################################


#Initialize all host data
f0_h = faceq1*rhoOut*(1 - 1.5*vxIn**2)                       *np.ones([nHeight,nWidth], dtype=np.float32)
f1_h = faceq2*rhoOut*(1 - 1.5*vxIn**2 + 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)
f2_h = faceq2*rhoOut*(1 - 1.5*vxIn**2)                       *np.ones([nHeight,nWidth], dtype=np.float32)
f3_h = faceq2*rhoOut*(1 - 1.5*vxIn**2 - 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)
f4_h = faceq2*rhoOut*(1 - 1.5*vxIn**2)                       *np.ones([nHeight,nWidth], dtype=np.float32)
f5_h = faceq3*rhoOut*(1 - 1.5*vxIn**2 + 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)
f6_h = faceq3*rhoOut*(1 - 1.5*vxIn**2 - 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)
f7_h = faceq3*rhoOut*(1 - 1.5*vxIn**2 - 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)
f8_h = faceq3*rhoOut*(1 - 1.5*vxIn**2 + 3*vxIn + 4.5*vxIn**2)*np.ones([nHeight,nWidth], dtype=np.float32)

g0_h = faceq1      *initialGradient()
g1_h = faceq2*(1 + 3*vxIn)      *initialGradient()
g2_h = faceq2      *initialGradient()
g3_h = faceq2*(1 - 3*vxIn)      *initialGradient()
g4_h = faceq2      *initialGradient()
g5_h = faceq3*(1 + 3*vxIn)      *initialGradient()
g6_h = faceq3*(1 - 3*vxIn)      *initialGradient()
g7_h = faceq3*(1 - 3*vxIn)      *initialGradient()
g8_h = faceq3*(1 + 3*vxIn)      *initialGradient()

solid_h = np.ones([nHeight, nWidth], dtype=np.int32)
solid_h[12,12] = 0 

#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
f0_d = gpuarray.to_gpu( f0_h )
f1_d = gpuarray.to_gpu( f1_h )
f2_d = gpuarray.to_gpu( f2_h )
f3_d = gpuarray.to_gpu( f3_h )
f4_d = gpuarray.to_gpu( f4_h )
f5_d = gpuarray.to_gpu( f5_h )
f6_d = gpuarray.to_gpu( f6_h )
f7_d = gpuarray.to_gpu( f7_h )
f8_d = gpuarray.to_gpu( f8_h )

g0_d = gpuarray.to_gpu( g0_h )
g1_d = gpuarray.to_gpu( g1_h )
g2_d = gpuarray.to_gpu( g2_h )
g3_d = gpuarray.to_gpu( g3_h )
g4_d = gpuarray.to_gpu( g4_h )
g5_d = gpuarray.to_gpu( g5_h )
g6_d = gpuarray.to_gpu( g6_h )
g7_d = gpuarray.to_gpu( g7_h )
g8_d = gpuarray.to_gpu( g8_h )

solid_d = gpuarray.to_gpu( solid_h )

#For textures 
#f0_dArr, copyf0_dTod = gpuArray2DtocudaArray( f0_d ) 
f1_dArr, copyf1_dTod = gpuArray2DtocudaArray( f1_d )
f2_dArr, copyf2_dTod = gpuArray2DtocudaArray( f2_d )
f3_dArr, copyf3_dTod = gpuArray2DtocudaArray( f3_d )
f4_dArr, copyf4_dTod = gpuArray2DtocudaArray( f4_d )
f5_dArr, copyf5_dTod = gpuArray2DtocudaArray( f5_d )
f6_dArr, copyf6_dTod = gpuArray2DtocudaArray( f6_d )
f7_dArr, copyf7_dTod = gpuArray2DtocudaArray( f7_d )
f8_dArr, copyf8_dTod = gpuArray2DtocudaArray( f8_d )

#g0_dArr, copyg0_dTod = gpuArray2DtocudaArray( g0_d ) 
g1_dArr, copyg1_dTod = gpuArray2DtocudaArray( g1_d )
g2_dArr, copyg2_dTod = gpuArray2DtocudaArray( g2_d )
g3_dArr, copyg3_dTod = gpuArray2DtocudaArray( g3_d )
g4_dArr, copyg4_dTod = gpuArray2DtocudaArray( g4_d )
g5_dArr, copyg5_dTod = gpuArray2DtocudaArray( g5_d )
g6_dArr, copyg6_dTod = gpuArray2DtocudaArray( g6_d )
g7_dArr, copyg7_dTod = gpuArray2DtocudaArray( g7_d )
g8_dArr, copyg8_dTod = gpuArray2DtocudaArray( g8_d )

tex_f1.set_array( f1_dArr )
tex_f2.set_array( f2_dArr )
tex_f3.set_array( f3_dArr )
tex_f4.set_array( f4_dArr )
tex_f5.set_array( f5_dArr )
tex_f6.set_array( f6_dArr )
tex_f7.set_array( f7_dArr )
tex_f8.set_array( f8_dArr )

tex_g1.set_array( g1_dArr )
tex_g2.set_array( g2_dArr )
tex_g3.set_array( g3_dArr )
tex_g4.set_array( g4_dArr )
tex_g5.set_array( g5_dArr )
tex_g6.set_array( g6_dArr )
tex_g7.set_array( g7_dArr )
tex_g8.set_array( g8_dArr )
#Memory for plotting
plotData_d = gpuarray.to_gpu(np.zeros_like(f0_h))
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 

def stepFunction():
  stream()
  collideKernel( np.int32(nWidth), np.int32(nHeight), beta, tau, tauT, faceq1, faceq2, faceq3,
		 f0_d, f1_d, f2_d, f3_d, f4_d, f5_d, f6_d, f7_d, f8_d, 
		 g0_d, g1_d, g2_d, g3_d, g4_d, g5_d, g6_d, g7_d, g8_d,
		 g, tempMin, tempMax, plotVar, plotData_d, grid=grid2D, block=block2D )
  applyBCsKernel ( np.int32(nWidth), np.int32(nHeight), vxIn, rhoOut, faceq1, faceq2, faceq3, 
		   f0_d, f1_d, f2_d, f3_d, f4_d, f5_d, f6_d, f7_d, f8_d, 
		   g0_d, g1_d, g2_d, g3_d, g4_d, g5_d, g6_d, g7_d, g8_d,
		   tempMin, tempMax, solid_d, grid=grid2D, block=block2D )
  applyPeriodicBCsKernel( np.int32(nWidth), np.int32(nHeight),
			   f1_d, f2_d, f3_d, f4_d, f5_d, f6_d, f7_d, f8_d,
			   g1_d, g2_d, g3_d, g4_d, g5_d, g6_d, g7_d, g8_d,
			   grid=grid2D, block=block2D )
			   

#configure animation2D stepFunction and plotData
animation2D.stepFunc = stepFunction
animation2D.plotData_d = plotData_d
animation2D.background_h = solid_h
animation2D.background_d = solid_d
animation2D.maxVar = np.float32( 1.1*tempMax )
animation2D.minVar = tempMin

#nIterations = 10000
#startTime = time.time()
#[stepFunction() for i in range(nIterations)]
#endTime = time.time()
#print "Time i {0} iterations: {1} secs".format(nIterations, endTime-startTime)
#print " Iter per second: {0}".format(nIterations/(endTime-startTime))



#run animation
animation2D.animate()

