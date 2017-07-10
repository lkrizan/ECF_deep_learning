# README #

Extension of [ECF](http://ecf.zemris.fer.hr/) (Evolutionary Computation Framework) for deep neural networks with TensorFlow r1.0 on MS Windows.


## REQUIREMENTS ##

* MS Visual Studio 2015 Update 2 or later versions
* TensorFlow r1.0


## Setting up TensorFlow ##

More information can be found in TensorFlow [readme](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/README.md).
### Pre-requisites ###
* [Git](http://git-scm.com)
* [SWIG](http://www.swig.org/download.html) (recommended version swigwin-3.0.10)
* [Python 3](https://www.continuum.io/downloads) (preferably Anaconda)
* [CMake](https://cmake.org/files/v3.6/cmake-3.6.3-win64-x64.msi) (recommended version 3.6.3)
* if using GPU:
    - [NVidia CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
    - [NVidia CUDNN 5.1](https://developer.nvidia.com/cudnn)
### Installing pre-requisites ###
#### SWIG ####
Download the zip archive and extract it to `C:\tools\` and add `C:\tools\` to `PATH` environment variable.
#### NVidia CUDNN ####
Download the zip archive after installing CUDA Toolkit and extract it to `C:\local\`. Add `C:\local\cuda` to your `PATH`.
#### Additional notes ####
Make sure that `cmake` and `git` are installed and in your `PATH`. Check if there's `CUDA_PATH` in your system variables.


### TensorFlow build ###

* Setup environment.
    - run cmd.exe and the following command:

```
C:\temp> "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
```

* Clone the TensorFlow repository, checkout r1.0 branch and create a working directory for your build:

```
C:\temp> git clone -b r1.0 --single-branch https://github.com/tensorflow/tensorflow.git
C:\temp> cd tensorflow\tensorflow\contrib\cmake
C:\temp\tensorflow\tensorflow\contrib\cmake> mkdir build
C:\temp\tensorflow\tensorflow\contrib\cmake> cd build
C:\temp\tensorflow\tensorflow\contrib\cmake\build>
```

* Invoke CMake to create Visual Studio solution and project files.
    - If you are building without GPU support, do not use the last two lines (and remove the `^` character (line continuation) from the last line.    
```
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=C:\tools\swigwin-3.0.10\swig.exe ^
-DPYTHON_EXECUTABLE=C:\Users\%USERNAME%\Anaconda3\python.exe ^
-DPYTHON_LIBRARIES=C:\Users\%USERNAME%\Anaconda3\libs\python35.lib ^
-DPYTHON_INCLUDE_DIR=C:\Users\%USERNAME%\Anaconda3\include ^
-DNUMPY_INCLUDE_DIR=C:\Users\%USERNAME%\Anaconda3\Lib\site-packages\numpy\core\include ^
-Dtensorflow_ENABLE_GPU=ON ^
-DCUDNN_HOME=C:\local\cuda\
```

* Invoke MSBuild to build TensorFlow. 
    - Run the following line:  

```
C:\...\build> MSBuild /p:Configuration=Release tf_tutorials_example_trainer.vcxproj
```    

    - Be patient, this may take few hours.
    - After the build is finished, test it:
    
```
C:\...\build> Release\tf_tutorials_example_trainer.exe
```
    
    
## Setting up project files ##

* all include paths in the DeepModelTraining.vcxproj are set over user-defined macro in Visual Studio, so after loading the solution, do the following:
    - select View > Property manager
    - select Release | x64 and right click on property ending with `.user`, then select Properties
    - under User Macro in common properties, add TENSORFLOW_ROOT, ECF_ROOT and BOOST_ROOT variables with value of location of their root directories
    - make sure you set option `Set this macro as an environment variable in the build environment`
    
* building ECF:
    - in ECF_lib.sln, set build configuration to Release x64
    - in solution properties, under C/C++ > Code Generation, set Runtime Library option
      to `Multi-threaded DLL` instead of `Multi-threaded`
    - optional: enable parameterization of multiple algorithms in ECF (used for hybrid approach which combines Backpropagation with other metaheuristic algorithms):
        - add the following lines to State::parseAlgorithmNode method in State.cpp file, just before final return true statement:

```
#!c++

for (int i = 1; i < n; ++i)
{
    XMLNode tmpNode = node.getChildNode(i);
    registry_->readEntries(tmpNode, tmpNode.getName());
}
```
      
* stop TensorFlow log messages (optional):
    - set environment variable `TF_CPP_MIN_LOG_LEVEL` with value 2 (this disables warnings and debug log)
    - to do it from Visual Studio, add `TF_CPP_MIN_LOG_LEVEL=2` to Configuration properties > Debugging > Environment in DeepModelTraining project properties
        
        
## Release notes ##

* release 1.0 includes GPU version (r1.0-gpu branch) and non-GPU version (r1.0-no-gpu branch)
* known limitation: PaddedMaxPool layer does not work on non-gpu version because TensorFlow does not support MaxPoolWithArgmax operation without GPU