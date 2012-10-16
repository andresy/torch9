SET(src)

SET(luasrc init.lua Timer.lua Storage.lua Tensor.lua apply.lua iterator.lua)
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
