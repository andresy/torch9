Torch9 Library.
===============

Torch9 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to an easy and fast scripting language (Luajit) and
few C code lines for critical inner loops.

Note: Torch9 is still beta.

Installation
------------

### Requirements

*   C compiler
*   [luajit](http://www.luajit.org)
*   [cmake](http://www.cmake.org)
*   git
*   [luarocks](http://www.luarocks.org)

### Getting the last version from the git:

```sh
luarocks build https://raw.github.com/andresy/torch9/master/rocks/torch-git-0.rockspec
```

Running
=======
```sh
$ luajit -ltorch
Torch 9.0 -- Copyright (C) 2001-2013 Idiap, NEC Labs, NYU. http://www.torch.ch/
LuaJIT 2.0.0 -- Copyright (C) 2005-2012 Mike Pall. http://luajit.org/
JIT: ON CMOV SSE2 SSE3 SSE4.1 fold cse dce fwd dse narrow loop abc sink fuse
> 
```

Documentation
=============

Coming soon.


