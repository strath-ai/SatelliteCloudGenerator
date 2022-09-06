# LocalGaussianBlur
PyTorch implementation of a Gaussian Blur operation with changing kernel variance

The operation has similar interface to the scripts in `torchvision`:

```python
lblur = LocalGaussianBlur()

blurred = lblur(image,modulator)
```

See the jupyter notebook for a working example.
