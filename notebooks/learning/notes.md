# Torch fun.s
1. torch.arange()
2. torch.zeros
3. torch.ones
4. torch.zeros_like(other_tensor)
5. torch.float16 -> and other dtypes
6. torch.matmul / torch.add / torch.subtract ....
7. torch.min(tensor_x)
8. torch.from_numpy
9. torch.manual_seed
9. torch.cuda.manual_seed



# 00_PYTorch_Fundamentals

## Data Types
### To know what is the datatype of 
> x is a tensor

`x.dtype()`

### To  set a data type for a tensor
`x = torch.tensor(dtype=torch.float32)`

### To change the data type
`x.type(torch.float32)`

`x.float()`


## Range
### In Tensors 
to create a range in a tensor 

`torch.arange()` 

### In for loop
`for i in range(10)`


## Tensors operations
### Addition
`torch.add(x, 10)`

`x + 10`

### Matrix Multiplication
`torch.matmul(x, x)`


## Aggregation  
### min
`x.min()`

`torch.min(x)`

### Find the position of the min
`x.argmin()`

`torch.argmin(x)`


## Manipulating a Tensor
### Reshaping to a compatible shape
`x.reshape([])`

### Remove all 1 dimensions
`x.squeeze()`

### Add extra dimension
`x.unsqueeze(dim=0)` What is the place of the dimension you want to add 
> if x size was (9) -> (1,9)

### Change the order of dimensions
`x.permute()` put the place of dimensions e.g., `(2, 0, 1)`



## Convert between tensors and numPy arrays
### From tensors to NumPy
`x.numpy()`

### From numPy to tensor
`torch.from_numpy(numpy_array)` 


## Reproducibility in randomness
### For CPU and general PyTorch randomness
RANDOM_SEED = a number
`torch.manual_seed(RANDOM_SEED)`

### For CUDA GPU randomness
`torch.cuda.manual_seed(RANDOM_SEED)`


## Tensors on GPU/CPU
### Create a tensor in a specific device
`x = torch.tensor(device='cuda')`

### Change the device of a tensor
`x.to('cpu')` // or `cuda`

`x.cpu()` // or `cuda`


# 01_PyTorch_Workflow

> `model` is an instance of out nn model 

## Model Moods
### 1. Training mode
`model.training()`

### 2. Evaluation mode
Used when testing the model or making predictions.
```
    # This is to make a prediction
    with torch.inference_mode():
        y_predictions = model(input_test_data)
```

> We can do it without the ` with torch.inference_mode():`, but adding it 
> reduces the computations via disabling tracking the gradients

`model.eval()`

when we want to evaluate the model inside the training loop 