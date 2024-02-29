## Error when installing 'gym==0.19.0'

```bash
wheel.vendored.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
    opencv-python>=3.
                ~~~^
[end of output]
```

Change the version of `wheel` with:

```bash
pip install wheel==0.38.4
```

## Missing file when using 'torchsparse'

```bash
error: google/dense_hash_map: No such file or directory
```

Install libsparsehash-dev with:

```bash
sudo apt-get update; sudo apt-get install libsparsehash-dev
```
