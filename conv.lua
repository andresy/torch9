local ffi = require 'ffi'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local C = require 'torch.TH'
local register_ = require 'torch.registernumbers'

-- handle method/function
local function register(args)
   if args.nomethod and not args.nofunction then
      return register_(args, torch, nil)
   elseif args.nofunction and not args.nomethod then
      return register_(args, nil, class.metatable('torch.RealTensor'))
   else
      return register_(args, torch, class.metatable('torch.RealTensor'))
   end
end

register{
   name = "conv2",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={opt=true}},
   {name="src2", type="torch.RealTensor"},
   {name="opt", type="string", default='V'},
   call =
      function(dst, src1, src2, opt)
         assert(opt == 'F' or opt == 'V', 'option must be F or V')
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         if src1.__nDimension == 2 and src2.__nDimension == 2 then
            C.THRealTensor_conv2Dmul(res, 0, 1, src1, src2, 1, 1, opt, 'C')
         elseif src1.__nDimension == 3 and src2.__nDimension == 3 then
            C.THRealTensor_conv2Dcmul(res, 0, 1, src1, src2, 1, 1, opt, 'C')
         elseif src1.__nDimension == 3 and src2.__nDimension == 4 then
            C.THRealTensor_conv2Dmv(res, 0, 1, src1, src2, 1, 1, opt, 'C')
         else
            error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
         end
         return res
      end
}

register{
   name = "xcorr2",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={opt=true}},
   {name="src2", type="torch.RealTensor"},
   {name="opt", type="string", default='V'},
   call =
      function(dst, src1, src2, opt)
         assert(opt == 'F' or opt == 'V', 'option must be F or V')
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         if src1.__nDimension == 2 and src2.__nDimension == 2 then
            C.THRealTensor_conv2Dmul(res, 0, 1, src1, src2, 1, 1, opt, 'X')
         elseif src1.__nDimension == 3 and src2.__nDimension == 3 then
            C.THRealTensor_conv2Dcmul(res, 0, 1, src1, src2, 1, 1, opt, 'X')
         elseif src1.__nDimension == 3 and src2.__nDimension == 4 then
            C.THRealTensor_conv2Dmv(res, 0, 1, src1, src2, 1, 1, opt, 'X')
         else
            error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
         end
         return res
      end
}

register{
   name = "conv3",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={opt=true}},
   {name="src2", type="torch.RealTensor"},
   {name="opt", type="string", default='V'},
   call =
      function(dst, src1, src2, opt)
         assert(opt == 'F' or opt == 'V', 'option must be F or V')
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         if src1.__nDimension == 3 and src2.__nDimension == 3 then
            C.THRealTensor_conv3Dmul(res, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
         elseif src1.__nDimension == 4 and src2.__nDimension == 4 then
            C.THRealTensor_conv3Dcmul(res, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
         elseif src1.__nDimension == 4 and src2.__nDimension == 5 then
            C.THRealTensor_conv3Dmv(res, 0, 1, src1, src2, 1, 1, 1, opt, 'C')
         else
            error('invalid source dimensions (expected: 2/2 or 3/3 or 3/4')
         end
         return res
      end
}

register{
   name = "xcorr3",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={opt=true}},
   {name="src2", type="torch.RealTensor"},
   {name="opt", type="string", default='V'},
   call =
      function(dst, src1, src2, opt)
         assert(opt == 'F' or opt == 'V', 'option must be F or V')
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         if src1.__nDimension == 3 and src2.__nDimension == 3 then
            C.THRealTensor_conv3Dmul(res, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
         elseif src1.__nDimension == 4 and src2.__nDimension == 4 then
            C.THRealTensor_conv3Dcmul(res, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
         elseif src1.__nDimension == 4 and src2.__nDimension == 5 then
            C.THRealTensor_conv3Dmv(res, 0, 1, src1, src2, 1, 1, 1, opt, 'X')
         else
            error('invalid source dimensions (expected: 3/3 or 4/4 or 4/5')
         end
         return res
      end
}
