# Neural Network From Scratch

## Run Locally
<!-- Say project is setup with cmake and give link -->
The project is setup with [cmake](https://cmake.org/). To run the project locally, follow the steps below:

### Clone the repository

```bash
  git https://github.com/Prateek61/neural-network-from-scratch.git
  cd neural-network-from-scratch
```

### Run cmake
```bash
  cmake -S . -B build
```
> Note: The build system can be changed by specifying the generator flag. For example, to use MinGW MakeFiles build system, append `-G "MinGW Makefiles"` to the above cmake command.

> Note: If you also want to run the tests, append `-D BUILD_TESTS=ON` to the above cmake command.

### Compile and run
```bash
  cmake --build build --target run_program
```

### Run Tests
```bash
  cmake --build build --target run_tests
```
