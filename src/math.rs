use std::f32::consts::PI;

use rand::prelude::*;
use vek::vec;

use crate::spectrum::Spectrum;

pub type Vec4 = vec::repr_c::Vec4<f32>;
pub type Vec3 = vec::repr_c::Vec3<f32>;
pub type Vec2 = vec::repr_c::Vec2<f32>;

pub type Mat3 = vek::mat::repr_c::Mat3<f32>;
pub type CVec3<T> = vec::repr_c::Vec3<T>;

pub type Quat = vek::quaternion::repr_c::Quaternion<f32>;

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub orientation: Quat,
}

pub trait OrthonormalBasis: Sized {
    fn get_orthonormal_basis(&self) -> Mat3;
}

impl OrthonormalBasis for Vec3 {
    fn get_orthonormal_basis(&self) -> Mat3 {
        let nor = *self;
        let ks = nor.z.signum();
        let ka = 1.0 / (1.0 + nor.z.abs());
        let kb = -ks * nor.x * nor.y * ka;
        let uu = Vec3::new(1.0 - nor.x * nor.x * ka, ks*kb, -ks*nor.x);
        let vv = Vec3::new(kb, ks - nor.y * nor.y * ka * ks, -nor.y);
        Mat3 {
            cols: CVec3::new(uu, vv, nor),
        }
    }
}

pub trait RandomSample<T> {
    fn rand_in_unit_sphere(rng: &mut ThreadRng) -> Self;
    fn rand_on_unit_sphere(rng: &mut ThreadRng) -> Self;
    fn cosine_weighted_in_hemisphere(rng: &mut ThreadRng, factor: T) -> Self;
}

impl RandomSample<f32> for Vec3 {
    fn rand_in_unit_sphere(rng: &mut ThreadRng) -> Self {
        let theta = rng.gen_range(0f32, 2f32 * PI);
        let phi = rng.gen_range(-1f32, 1f32);
        let ophisq = (1.0 - phi * phi).sqrt();
        Vec3::new(ophisq * theta.cos(), ophisq * theta.sin(), phi)
    }

    fn rand_on_unit_sphere(rng: &mut ThreadRng) -> Self {
        Self::rand_in_unit_sphere(rng).normalized()
    }

    fn cosine_weighted_in_hemisphere(rng: &mut ThreadRng, constriction: f32) -> Self {
        (Vec3::unit_z() + Self::rand_on_unit_sphere(rng) * constriction).normalized()
    }
}

pub fn f0_from_ior(ior: f32) -> f32 {
    let f0 = (1.0 - ior) / (1.0 + ior);
    f0 * f0
}

pub fn f_schlick(cos: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos).powi(5)
}

pub fn f_schlick_c(cos: f32, f0: Spectrum) -> Spectrum {
    let f0 = f0.0;
    let out_v = f0 + (Vec3::one() - f0) * (1.0 - cos).powi(5);
    Spectrum(out_v)
}

pub fn saturate(v: f32) -> f32 {
    v.min(1.0).max(0.0)
}
