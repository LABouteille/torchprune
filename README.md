# torchcompress

- A deep learning compression framework in Pytorch (work in progress) 
    - Support Filter pruning:
        - Conv -> Conv
        - Linear -> Linear
        - Conv -> Flatten -> Linear
        - Batchnorm2d/1d
        
# To do

- [ ] Pruning
    - [ ] Unstructured pruning
    - [ ] Structured pruning
- [ ] Quantization
    - [ ] Post Training Quantization
    - [ ] Quantization aware training
- [ ] Knowledge distillation
   - [ ] Offline distillation
   - [ ] Online distillation
   - [ ] Self distillation
