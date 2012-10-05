SET(src)

SET(luasrc init.lua Storage.lua Tensor.lua)
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
