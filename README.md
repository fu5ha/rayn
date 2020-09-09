# rayn

This is a CPU-based path tracing renderer focused on rendering SDFs, specifically fractals. It was originally based on the book "Ray tracing in one weekend" by Peter Shirley, which I heartily recommend, though it has now evolved into a structure of my own design, taking hints from that, `pbrt`, which is also an excellent reference, and research from NVIDIA on [wavefront pathtracing](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus) for taking advantage of SIMD.

# Features

* Architected to use 128-wide SIMD to full extent with the help of [`ultraviolet`](https://github.com/termhn/ultraviolet), and in the future perhaps 256 or 512 as well.
* Physical light transport algorithm
* Multiple-bounce indirect lighting/global illumination
* Importance sampling (soon multiple importance sampling)
* Next Event Estimation / Direct light sampling
* Depth of field
* Arbitrary animation and time-sampled motion blur
* Signed distance field rendering through leveraging [`sdfu`](https://github.com/termhn/sdfu/)
* Homogeneous volumetrics with extinction and single scattering with isotropic media

## Demo images

*All demo images in this repository are licensed under the **CC BY-NC-ND** license which essentially means you are free to use them for **non-commercial purposes** so long as you:*

*1. Credit me (my name, Gray Olson, and a link to my website, https://grayolson.me/)*

*2. Do not modify them*

![[CC BY-NC-ND License Badge](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-nd.svg)

*If you wish to use them for commercial purposes, contact me and we can work out a license.*


![demo7](/render7.jpg?raw=true)
![demo1](/render1.png?raw=true)
*Full 8k resolution image of this render [available here](https://live.staticflickr.com/65535/49550233828_4a967c0d7c_o_d.png).*

# How to use it

## Building

First, [install Rust](https://doc.rust-lang.org/book/ch01-01-installation.html). Then, clone or download this repo (green "Code" button on GitHub).
Finally, open a shell prompt and change to this repository's directory (if you're not sure how to do this, see
[this article](https://www.hongkiat.com/blog/web-designers-essential-command-lines/)). Then, run

```
$ cargo run --release
```

This will render an image and place it in a folder called `renders` inside the folder this repo is in.

### Playing with the scene

With your favorite code editor (I recommend [VSCode](https://code.visualstudio.com/) with the `rust-analyzer` plugin), open the `src/setup.rs` file.

Here you can change many settings including the resolution of the output image, the number of indirect lighting bounces, the number of raymarching steps
on each path, the number of total samples to take, and the setup of the whole scene. Feel free to play with all these numbers and see what they do. Just
run `cargo run --release` each time you make a change to render a new image. There are some comments in that file to help you get started on things you
can play around with.

## More demo images

![demo2](/render2.png?raw=true)
![demo3](/render3.png?raw=true)
![demo4](/render4.png?raw=true)
![demo5](/render5.png?raw=true)
![demo6](/render6.png?raw=true)
