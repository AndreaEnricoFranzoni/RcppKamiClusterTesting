# RcppKamiClusterTesting

# Prerequisites

R has to be updated at least to 4.0.0 version. If Windows is used, R version has to be at least 4.4.0.

On R console:
~~~
library(devtools)
~~~

Or, alternatively, if not installed:
~~~
install.packages("devtools")
library(devtools)
~~~

**`RcppKamiClusterTesting`** depends also on having Fortran, Lapack, BLAS and OpenMP installed. For Linux and Windows, GCC compiler version needed is 13.0.0.. On the other hand, for macOS, clang compiler version has to be at least 19.0.0.. Depending on the operative system, the instructions to set up everything can be found [here below](#prerequisites-depending-on-operative-system).

C++ version used is c++20 (the most recent within the stable versions used by Rcpp).




# Installation

To install the package, depending on the operative system:

- **Linux**
~~~
devtools::install_github("AndreaEnricoFranzoni/RcppKamiClusterTesting")
~~~

- **Windows**
~~~
devtools::install_github("AndreaEnricoFranzoni/RcppKamiClusterTesting")
~~~

- **macOS**
~~~
install.packages("withr")
library(withr)
~~~

and consequently, depending on the processor:

  - Intel processor
    ~~~
    withr::with_path(
        new = "/usr/local/opt/llvm/bin",
        devtools::install_github("AndreaEnricoFranzoni/RcppKamiClusterTesting")
    )
    ~~~

  - Apple processor (M1/M2/M3)
    ~~~
    withr::with_path(
        new = "/opt/homebrew/opt/llvm/bin",
        devtools::install_github("AndreaEnricoFranzoni/RcppKamiClusterTesting")
    )
    ~~~




Upload the library in the R environment
~~~
library(RcppKamiClusterTesting)
~~~



Due to the high number of warnings, to disable them can be useful adding as argument of `install_github`
~~~
quiet=TRUE
~~~


If problem related to dependencies arises when installing, also the argument 
~~~
dependencies = TRUE
~~~
could be useful



# Prerequisites: depending on operative system

More detailed documentation can be found in [this section](https://cran.r-project.org) of `The R Manuals`.
Although installing **`RcppKamiClusterTesting`** should automatically install all the R dependecies, could be worth trying to install them manaully if an error occurs.
~~~
install.packages("Rcpp")
install.packages("RcppEigen")
library(Rcpp)
library(RcppEigen)
~~~

## macOS

1. **Fortran**:  unlike Linux and Windows, Fortran has to be installed on macOS: instructions in this [link](https://cran.r-project.org/bin/macosx/tools/). Lapack and BLAS will be consequently installed.

2. **clang and OpenMP**: `Apple clang` version has to be at least 19.0.0.. Unlike Linux and Windows, OMP is not installed by default on macOS. Open the terminal and digit the following commands.

- **Homebrew**
  - Check the presence of Homebrew
    ~~~
    brew --version
    ~~~
    - If this command does not give back the version of Homebrew, install it according to the macOS architecture 
    ~~~
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ~~~
      1. *M1*, *M2* or *M3*
      ~~~
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
      ~~~
      2. *Intel*
      ~~~
      echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
      eval "$(/usr/local/bin/brew shellenv)"
      ~~~

- **LLVM/clang**
   
  [LLVM](https://llvm.org) toolchain is needed to configure clang on macOS in order to use **`fdaPDE`** (external library needed by **`RcppKamiClusterTesting`**) and an external OMP version
  - Check its presence. 
  ~~~
  llvm-config --version
  ~~~
  - `Apple clang` version has to be at least 19.0.0.. Check it via:
    1. *M1*, *M2* or *M3*
      ~~~
      /opt/homebrew/opt/llvm/bin/clang++ --version
      ~~~
      2. *Intel*
      ~~~
      /usr/local/opt/llvm/bin/clang++ --version
      ~~~
  - Download it if not present or `Apple clang` version is not enough recent
  ~~~
  brew install llvm
  ~~~
  

- **OMP**
  - Once Homebrew is set, check the presence of OMP
    ~~~
    brew list libomp
    ~~~
  - Install it in case of negative output
    ~~~
    brew install libomp
    ~~~




## Windows

- **Rtools**: can be installed from [here](https://cran.r-project.org/bin/windows/Rtools/). Version 4.4 is needed to install parallel version.


## Linux

- Linux installation depends on its distributor. All the commands here reported will refer to Ubuntu and Debian distributors. Standard developement packages have to be installed.   In Ubuntu and Debian, for example, all the packages have been collected into a single one, that is possible to install digiting into the terminal:

   ~~~
  sudo apt install r-base-dev
  sudo apt install build-essential
   ~~~

## Linux image
Before being able to run the commands explained above for Linux, R has to be downloaded. Afterward, the installation of Fortran, Lapack, BLAS, devtools and its dependecies can be done by digiting into the terminal:
   ~~~
sudo apt-get update
sudo apt install gfortran
sudo apt install liblapack-dev libblas-dev
   ~~~
   ~~~
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libssl-dev
sudo apt-get install libz-dev
sudo apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
sudo apt install zlib1g-dev
sudo apt install -y libfreetype6-dev libfontconfig1-dev libharfbuzz-dev libcairo2-dev libpango1.0-dev pandoc
   ~~~
