# rayn

This is a CPU-based path tracing renderer focused on rendering SDFs, specifically fractals. It was originally based on the book "Ray tracing in one weekend" by Peter Shirley, which I heartily recommend, though it has now evolved into a structure of my own design, taking hints from that, `pbrt`, which is also an excellent reference, and research from NVIDIA on [wavefront pathtracing](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus) for taking advantage of SIMD.


# Features

* Architected to use 128-wide SIMD to full extent with the help of [`ultraviolet`](https://github.com/termhn/ultraviolet), and in the future perhaps 256 or 512 as well.
* Physical light transport algorithm
* Importance sampling (soon multiple importance sampling)
* Next Event Estimation / Direct light sampling
* Depth of field
* Arbitrary animation and time-sampled motion blur
* Multiple-bounce indirect lighting/global illumination
* Signed distance field rendering through leveraging [`sdfu`](https://github.com/termhn/sdfu/)

### Demo images

*All demo images in this repository are licensed under the **CC BY-NC-ND** license which essentially means you are free to use them for **non-commercial purposes** so long as you:*

*1. Credit me (my name, Gray Olson, and a link to my website, https://grayolson.me/)*

*2. Do not modify them*

![[CC BY-NC-ND License Badge](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-nd.svg)

*If you wish to use them for commercial purposes, contact me and we can work out a license.*


![demo7](/render7.jpg?raw=true)
![demo1](/render1.png?raw=true)
*Full 8k resolution image of this render [available here](https://live.staticflickr.com/65535/49550233828_4a967c0d7c_o_d.png).*
![demo2](/render2.png?raw=true)
![demo3](/render3.png?raw=true)
![demo4](/render4.png?raw=true)
![demo5](/render5.png?raw=true)
![demo6](/render6.png?raw=true)
