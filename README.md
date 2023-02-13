# ailia Rust example
This repository is an example of using the ailia SDK from rust.
## Function
Performs object recognition on PC camera images.

## Requirements
* `opencv` 4.7.0 or later
* `rust` 1.56.0 or later
* `ailia SDK` 1.2.13 or later

## Usage
### Path configuration
for linux user
```bash
export AILIA_INC_DIR=[path/to/ailia.h]
export AIILA_BIN_DIR=[path/to/libailia.so]
export LD_LIBRARY_PATH=[path/to/libailia.so]:LD_LIBRARY_PATH
```
for mac user
```bash
export AILIA_INC_DIR=[path/to/ailia.h]
export AIILA_BIN_DIR=[path/to/libailia.dylib]
export DYLD_LIBRARY_PATH=[path/to/libailia.dylib]:DYLD_LIBRARY_PATH
```

`cargo run --release`