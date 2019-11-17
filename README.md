# rayn

This is a CPU-based path tracing renderer focused on rendering SDFs, specifically fractals. It was originally based on the book "Ray tracing in one weekend" by Peter Shirley, which I heartily recommend, though it has now evolved into a structure of my own design, taking hints from both that and `pbrt`, which is also an excellent reference.


# Features

* Architected to use 128-wide SIMD to full extent with the help of [`ultraviolet`](https://github.com/termhn/ultraviolet), and in the future perhaps 256 or 512 as well.
* Physical light transport algorithm
* Depth of field
* Arbitrary animation and time-sampled motion blur
* Importance sampling (soon multiple importance sampling)
* Multiple-bounce indirect lighting/global illumination
* Signed distance field rendering through leveraging [`sdfu`](https://github.com/termhn/sdfu/)

### Demo images

![demo1](/render1.png?raw=true)
![demo2](/render2.png?raw=true)
![demo3](/render3.png?raw=true)
![demo4](/render4.png?raw=true)