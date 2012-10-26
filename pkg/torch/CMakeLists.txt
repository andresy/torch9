SET(src)

SET(luasrc init.lua argcheck.lua argtypes.lua Timer.lua Storage.lua Tensor.lua apply.lua dimapply.lua iterator.lua)
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
