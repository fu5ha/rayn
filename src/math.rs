use crate::spectrum::WSrgb;

use std::f32::consts::PI;
pub use wide::f32x4;

use rand::rngs::SmallRng;
use rand::Rng;
use vek::vec;

pub type Vec3 = ultraviolet::Vec3;
pub type Wec3 = ultraviolet::Wec3;
pub type Vec2 = ultraviolet::Vec2;
pub type Wec2 = ultraviolet::Wec2;
pub type Vec2u = vec::repr_c::Vec2<usize>;
pub type Aabru = vek::geom::repr_c::Aabr<usize>;
pub type Extent2u = vek::vec::repr_c::Extent2<usize>;

pub type Wat3 = ultraviolet::Wat3;

#[derive(Clone, Copy)]
pub struct Transform {
    pub position: Wec3,
    // pub orientation: Quat,
}

pub trait OrthonormalBasis<M>: Sized {
    fn get_orthonormal_basis(&self) -> M;
}

impl OrthonormalBasis<Wat3> for Wec3 {
    fn get_orthonormal_basis(&self) -> Wat3 {
        let nor = *self;
        let ks = nor.z.signum();
        let ka = f32x4::from(1.0) / (f32x4::from(1.0) + nor.z.abs());
        let kb = -ks * nor.x * nor.y * ka;
        let uu = Wec3::new(f32x4::from(1.0) - nor.x * nor.x * ka, ks * kb, -ks * nor.x);
        let vv = Wec3::new(kb, ks - nor.y * nor.y * ka * ks, -nor.y);
        Wat3::new(uu, vv, nor)
    }
}

pub trait RandomSample2d {
    fn rand_in_unit_disk(rng: &mut SmallRng) -> Self;
}

impl RandomSample2d for Wec2 {
    fn rand_in_unit_disk(rng: &mut SmallRng) -> Self {
        let rho = rng.gen::<[f32; 4]>();
        let rho = f32x4::from(rho).sqrt();
        let theta = rng.gen::<[f32; 4]>();
        let theta = f32x4::from(theta) * f32x4::from(2f32 * PI);
        Wec2::new(rho * theta.cos(), rho * theta.sin())
    }
}

pub trait RandomSample3d<T> {
    fn rand_in_unit_sphere(rng: &mut SmallRng) -> Self;
    fn rand_on_unit_sphere(rng: &mut SmallRng) -> Self;
    fn cosine_weighted_in_hemisphere(rng: &mut SmallRng, factor: T) -> Self;
}

impl RandomSample3d<f32x4> for Wec3 {
    fn rand_in_unit_sphere(rng: &mut SmallRng) -> Self {
        let theta = rng.gen::<[f32; 4]>();
        let theta = f32x4::from(theta) * f32x4::from(2f32 * PI);
        let phi = rng.gen::<[f32; 4]>();
        let phi = f32x4::from(phi) * f32x4::from(2.0) - f32x4::from(1.0);
        let ophisq = (f32x4::from(1.0) - phi * phi).sqrt();
        Wec3::new(ophisq * theta.cos(), ophisq * theta.sin(), phi)
    }

    fn rand_on_unit_sphere(rng: &mut SmallRng) -> Self {
        Self::rand_in_unit_sphere(rng).normalized()
    }

    fn cosine_weighted_in_hemisphere(rng: &mut SmallRng, constriction: f32x4) -> Self {
        let xy = Wec2::rand_in_unit_disk(rng) * constriction;
        let z = (f32x4::from(1.0) - xy.mag_sq()).sqrt();
        Wec3::new(xy.x, xy.y, z)
    }
}

#[allow(dead_code)]
pub fn f0_from_ior(ior: f32x4) -> f32x4 {
    let f0 = (f32x4::from(1.0) - ior) / (f32x4::from(1.0) + ior);
    f0 * f0
}

pub fn f_schlick(cos: f32x4, f0: f32x4) -> f32x4 {
    f0 + (f32x4::from(1.0) - f0) * (f32x4::from(1.0) - cos).powi([5, 5, 5, 5])
}

pub fn f_schlick_c(cos: f32x4, f0: WSrgb) -> WSrgb {
    f0 + (WSrgb::one() - f0) * (f32x4::from(1.0) - cos).powi([5, 5, 5, 5])
}

#[allow(dead_code)]
pub fn saturate(v: f32x4) -> f32x4 {
    v.min(f32x4::from(1.0)).max(f32x4::from(0.0))
}

pub struct CDF {
    items: Vec<(f32, f32)>,
    densities: Vec<f32>,
    weight_sum: f32,
    prepared: bool,
}

impl CDF {
    pub fn new() -> Self {
        CDF {
            items: Vec::new(),
            densities: Vec::new(),
            weight_sum: 0.0,
            prepared: false,
        }
    }

    pub fn insert(&mut self, item: f32, weight: f32) {
        self.items.push((item, weight));
        self.weight_sum += weight;
    }

    pub fn prepare(&mut self) {
        if self.prepared {
            return;
        }

        for (_, weight) in self.items.iter_mut() {
            *weight /= self.weight_sum;
        }

        let mut cum = 0.0;
        for (_, weight) in self.items.iter() {
            cum += *weight;
            self.densities.push(cum);
        }

        for (&(_, weight), density) in self.items.iter().zip(self.densities.iter_mut()).rev() {
            *density = 1.0;
            if weight > 0.0 {
                break;
            }
        }

        self.prepared = true;
    }

    pub fn sample(&self, x: f32) -> Option<(f32, f32)> {
        for (ret, density) in self.items.iter().zip(self.densities.iter()) {
            if *density >= x {
                return Some(*ret);
            }
        }
        None
    }
}
